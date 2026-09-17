"""Cross-component v2 scheduling, raw-caption, sampling and CLI contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys

import pytest
import torch
import yaml

from extensions.gen2_trainer.v2.config import resolve_process_config, SPEC_SHA256
from extensions.gen2_trainer.v2.process import V2Runner, prepare_batch, training_contract, specification_path


def config(tmp_path):
    return resolve_process_config({"training_folder": str(tmp_path), "trigger_word": "<test>",
        "model": {"name_or_path": "fixture"}, "datasets": [{"folder_path": "dataset"}],
        "sample": {"prompts": ["a tiger in [trigger] style"], "sample_every": 2},
        "train": {"steps": 4}, "save": {"save_every": 3},
        "gen2": {"schema_version": "2.0.0", "inference": {"unconditional_model_path": "fixture"},
                 "evaluation": {"named_phrase": "benchmark"}}})


class Recorder:
    def __init__(self): self.rows = []
    def set_context(self, **kwargs): pass
    def event(self, name, **kwargs): self.rows.append((name, kwargs))
    def record(self, name, row): self.rows.append((name, row))


def test_sampling_and_save_clocks_are_independent(tmp_path):
    cfg = config(tmp_path)
    runner = V2Runner(cfg, "smoke")
    sampled, saved = [], []
    runner.engine = SimpleNamespace(logical_update=0)
    runner.recorder = Recorder()
    runner.evaluation = SimpleNamespace(sample=sampled.append)
    runner.save = lambda **kw: saved.append((runner.engine.logical_update, kw["final"]))
    runner.events(initial=True)
    for update in range(1, 5):
        runner.engine.logical_update = update
        runner.events()
    assert sampled == [0, 2, 4]
    assert saved == [(3, False), (4, True)]


def test_disable_sampling_preserves_save_and_final(tmp_path):
    cfg = config(tmp_path); cfg["train"]["disable_sampling"] = True
    runner = V2Runner(cfg, "smoke")
    runner.engine = SimpleNamespace(logical_update=4)
    runner.recorder = Recorder()
    runner.evaluation = SimpleNamespace(sample=lambda *a: pytest.fail("sampling disabled"))
    saves = []; runner.save = lambda **kw: saves.append(kw)
    runner.events()
    assert saves == [{"final": True}]


def test_v2_prepared_data_retains_every_original_marker(tmp_path):
    cfg = config(tmp_path)
    path = str((tmp_path / "x.png").resolve())
    raw = '  {"style":"[trigger]", "desc":"[trigger] tiger, <test>"}  '
    source = {"sample_id": "x", "dataset_index": 0, "group": "all", "original_caption": raw}
    class DTO:
        latents = torch.ones(1, 1, 2, 2)
        file_items = [SimpleNamespace(path=path)]
        cleaned = False
        def cleanup(self): self.cleaned = True
    dto = DTO()
    model = SimpleNamespace(device_torch=torch.device("cpu"), torch_dtype=torch.float32,
        noise_scheduler=SimpleNamespace(timesteps=torch.tensor([1000., 500., 1.])),
        get_latent_noise_from_latents=lambda z, **kw: torch.zeros_like(z),
        add_noise=lambda z, noise, t: z*(1-t[:, None, None, None]/1000.))
    batch = prepare_batch(dto, model, cfg, {path: source})
    assert dto.cleaned and batch["qs"] == [raw]
    assert batch["metadata"][0]["q"] == raw
    torch.testing.assert_close(batch["target"], -torch.ones_like(dto.latents))


def test_version_dispatch_and_reference_identity(tmp_path):
    from extensions.gen2_trainer.__main__ import versioned_config, parser
    cfg = config(tmp_path)
    resolved, digest = versioned_config(cfg)
    assert digest == SPEC_SHA256
    assert resolved["_gen2_resolved"]["trainable_parameter_family"] == "embedding"
    assert hashlib.sha256(specification_path(resolved).read_bytes()).hexdigest() == SPEC_SHA256
    assert versioned_config({})[1] != SPEC_SHA256
    assert parser().parse_args(["infer", "package", "--prompt", "[trigger]", "--output", "x.png", "--mode", "learned"]).mode == "learned"


def test_resume_contract_changes_only_runtime_locations(tmp_path):
    first = config(tmp_path)
    second = deepcopy(first)
    second["name"] = "renamed"
    second["training_folder"] = str(tmp_path / "new")
    second["gen2"]["checkpoint"]["resume_from"] = "somewhere/resume_latest"
    assert training_contract(first) == training_contract(second)
    second["gen2"]["optimizer"]["eps"] = 1e-6
    assert training_contract(first) != training_contract(second)


def test_fixed_visual_references_reused_and_initial_alias(tmp_path, monkeypatch):
    from PIL import Image
    from extensions.gen2_trainer.v2 import evaluation
    cfg = config(tmp_path)
    class Compiler:
        def comparison(self, prompt, named_phrase):
            return {mode: mode for mode in ("base", "named", "init", "learned")}
    tokens = torch.nn.Linear(2, 2)
    backend = SimpleNamespace(tokens=tokens, compiler=Compiler())
    calls = []
    def generate(backend, prompt, mode, **kwargs):
        calls.append((mode, kwargs["seed"], kwargs["compiled"]))
        return Image.new("RGB", (16, 16), {"base": "black", "named": "blue", "init": "green", "learned": "red"}[mode]), {"mode": mode, "seed": kwargs["seed"]}
    monkeypatch.setattr(evaluation, "generate", generate)
    module = ModuleType("toolkit.config_modules")
    class Generate:
        def __init__(self, **kwargs): self.path = kwargs["output_path"]
        def save_image_atomic(self, image): image.save(self.path)
    module.GenerateImageConfig = Generate
    monkeypatch.setitem(sys.modules, "toolkit.config_modules", module)
    observer = evaluation.V2Evaluation(backend, cfg, Recorder(), tmp_path)
    torch.manual_seed(112); expected_rng = torch.random.get_rng_state().clone()
    observer.sample(0)
    assert len(calls) == 6 and not any(mode == "learned" for mode, _, _ in calls)
    assert len(observer.references) == 6
    assert (tmp_path / "samples/u00000000_comparison.png").is_file()
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    state = observer.state_dict()
    observer = evaluation.V2Evaluation(backend, cfg, Recorder(), tmp_path)
    observer.load_state_dict(state)
    observer.sample(2)
    assert calls[6:] == [("learned", 42, "learned"), ("learned", 43, "learned")]
    assert (tmp_path / "samples/u00000002_comparison.png").is_file()
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    reference = tmp_path / next(iter(state["references"].values()))["path"]
    reference.write_bytes(b"changed")
    with pytest.raises(ValueError, match="reference changed"):
        observer.sample(4)


def test_v2_examples_preserve_exact_user_prompt_and_spec_bytes():
    extension = Path(__file__).resolve().parents[1]
    for name, steps in (("smoke_gen2_ideogram4_v2.example.yaml", 4), ("pilot_gen2_ideogram4_v2.example.yaml", 500)):
        raw = yaml.safe_load((extension / "config" / name).read_text(encoding="utf-8"))["config"]["process"][0]
        resolved = resolve_process_config(raw)
        assert resolved["train"]["steps"] == steps
        prompt = resolved["sample"]["prompts"][0]
        assert prompt.count("[trigger]") == 3
        assert hashlib.sha256(prompt.encode()).hexdigest() == "b57f5ce3d890147803624284a2a3ff6b337dd1107033aedcc132b297eb5de17f"
