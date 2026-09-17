"""Configuration rejection boundaries and tracked VM examples for v2."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from extensions.gen2_trainer.v2.config import ConfigError, SPEC_SHA256, resolve_process_config


def minimal():
    return {
        "model": {"name_or_path": "ideogram-ai/ideogram-4-fp8"},
        "datasets": [{"folder_path": "/vm/dataset"}],
        "sample": {"prompts": ["A tiger in [trigger] style"]},
        "gen2": {"schema_version": "2.0.0", "inference": {
            "unconditional_model_path": "ideogram-ai/ideogram-4-fp8"}},
    }


def assign(config, dotted, value):
    parts = dotted.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def test_defaults_preserve_activator_ownership_and_effective_batch():
    raw = minimal()
    before = deepcopy(raw)
    cfg = resolve_process_config(raw)
    assert raw == before
    assert "network" not in cfg
    assert cfg["train"]["train_unet"] is False
    assert cfg["train"]["train_text_encoder"] is False
    assert cfg["train"]["max_grad_norm"] == 0
    assert cfg["_gen2_resolved"]["effective_batch_size"] == 4
    assert cfg["gen2"]["optimizer"] == {
        "type": "adamw", "lr": 5e-4, "eps": 1e-8,
        "betas": [0.9, 0.999], "weight_decay": 0,
    }
    assert cfg["save"]["max_step_saves_to_keep"] == 1
    assert cfg["gen2"]["expected_spec_sha256"] == SPEC_SHA256
    assert resolve_process_config(cfg) == cfg


def test_configuration_does_not_import_model_dependencies():
    code = (
        "import sys; from extensions.gen2_trainer.v2.config import resolve_process_config; "
        f"resolve_process_config({minimal()!r}); "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
        "assert 'diffusers' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path,value", [
    ("network", {"type": "lora"}),
    ("adapter", {"type": "lora"}),
    ("gen2.phases", {}),
    ("gen2.gates", {}),
    ("gen2.conditioning.adapter_rank", 4),
    ("gen2.conditioning.initializer_text", "Ghibli anime"),
    ("gen2.losses", {"neutral_weight": 1}),
    ("gen2.optimizers", {}),
    ("gen2.schema_version", "1.0.0"),
    ("gen2.execution.nonexistent", True),
    ("train.optimizer", "adamw"),
    ("train.optimizer_params", {"eps": 1e-8}),
    ("train.lr", 5e-4),
    ("train.lr_scheduler", "cosine"),
    ("train.train_unet", True),
    ("train.train_text_encoder", True),
    ("train.do_cfg", True),
    ("train.cfg_scale", True),
    ("train.cache_text_embeddings", True),
    ("train.unload_text_encoder", True),
    ("train.ema_config.use_ema", True),
    ("train.ema_config.unknown", False),
    ("train.skip_first_sample", True),
    ("train.noise_offset", 0.1),
    ("train.loss_type", "mae"),
    ("train.gradient_accumulation", 4),
    ("model.arch", "flux"),
    ("model.low_vram", True),
    ("model.layer_offloading", True),
    ("model.compile", True),
    ("model.lora_path", "/weights.safetensors"),
    ("model.unconditional_lora_path", "/correction.safetensors"),
    ("model.qtype", "qfloat8|recovery"),
    ("model.model_kwargs.unknown", 12),
    ("sample.walk_seed", True),
    ("sample.neg", "bad quality"),
    ("sample.sampler", "ddim"),
    ("sample.sample_start_step", 50),
    ("save.dtype", "bf16"),
    ("save.max_step_saves_to_keep", 15),
    ("save.push_to_hub", True),
    ("logging.use_wandb", True),
])
def test_rejects_options_that_change_or_bypass_the_declared_mechanism(path, value):
    raw = minimal()
    assign(raw, path, value)
    with pytest.raises(ConfigError):
        resolve_process_config(raw)


@pytest.mark.parametrize("path,value", [
    ("gen2.optimizer.lr", 0), ("gen2.optimizer.lr", float("inf")),
    ("gen2.optimizer.eps", float("nan")), ("gen2.optimizer.eps", False),
    ("gen2.optimizer.betas", [0.9]), ("gen2.optimizer.betas", [True, 0.99]),
    ("gen2.optimizer.betas", [0.9, 1.0]), ("gen2.optimizer.betas", "0.9,0.99"),
    ("gen2.optimizer.type", "automagic2"),
    ("gen2.optimizer.weight_decay", -1),
    ("gen2.conditioning.num_tokens", 0), ("gen2.conditioning.num_tokens", True),
    ("gen2.conditioning.initialization_sample_size", 1),
    ("gen2.conditioning.overflow_policy", "skip"),
    ("gen2.diagnostics.time_bin_edges", [0, 0.5, 0.4, 1]),
    ("gen2.evaluation.seeds", []), ("gen2.evaluation.seeds", [42, 42]),
    ("gen2.evaluation.seeds", [True]), ("gen2.evaluation.named_phrase", ""),
    ("gen2.evaluation.named_phrase", "[trigger] anime"),
    ("gen2.expected_spec_sha256", "a" * 64),
    ("train.steps", 0), ("train.batch_size", True),
    ("train.gradient_accumulation_steps", 0),
    ("train.gradient_checkpointing", "true"),
    ("train.num_train_timesteps", 1), ("train.max_grad_norm", -1),
    ("train.dtype", "fp16"), ("model.dtype", "float16"),
    ("model.model_kwargs.max_text_length", 3073),
    ("model.model_kwargs.max_text_length", 4),
    ("model.model_kwargs.max_text_length", 3072.0),
    ("sample.guidance_scale", float("inf")), ("sample.width", 513),
    ("sample.sample_every", 0), ("sample.sample_steps", False),
    ("sample.prompts", ["A tiger without an activator"]),
    ("sample.prompts", [{"prompt": "A tiger in [trigger] style"}]),
    ("sample.seed", 41),
])
def test_invalid_values_fail_before_model_load(path, value):
    raw = minimal()
    assign(raw, path, value)
    with pytest.raises(ConfigError):
        resolve_process_config(raw)


@pytest.mark.parametrize("entry", [
    {"cache_text_embeddings": True}, {"caption_dropout_rate": 0.1},
    {"shuffle_tokens": True}, {"is_reg": True}, {"num_workers": 2},
    {"flip_x": True}, {"replacements": ["cat|dog"]},
    {"resolution": []}, {"resolution": [512, 512]}, {"resolution": [513]},
    {"resolution": [False]}, {"num_repeats": 0}, {"loss_multiplier": 2},
])
def test_dataset_features_cannot_silently_change_the_objective_or_caption(entry):
    raw = minimal()
    raw["datasets"][0].update(entry)
    with pytest.raises(ConfigError):
        resolve_process_config(raw)


def test_sampling_and_save_intervals_are_independent_and_optimizer_can_be_8bit():
    raw = minimal()
    raw["save"] = {"save_every": 175}
    raw["sample"]["sample_every"] = 100
    raw["gen2"]["optimizer"] = {"type": "adamw8bit"}
    cfg = resolve_process_config(raw)
    assert cfg["sample"]["sample_every"] == 100
    assert cfg["save"]["save_every"] == 175
    assert cfg["gen2"]["optimizer"]["eps"] == 1e-8


def test_no_named_phrase_is_required_and_evaluation_seeds_control_first_seed():
    raw = minimal()
    raw["gen2"]["evaluation"] = {"seeds": [7, 19]}
    cfg = resolve_process_config(raw)
    assert cfg["gen2"]["evaluation"]["named_phrase"] is None
    assert cfg["sample"]["seed"] == 7


def test_dataset_and_model_sources_are_required_but_not_opened_by_resolver():
    raw = minimal()
    for section, key in (("model", "name_or_path"), ("gen2", "inference")):
        bad = deepcopy(raw)
        del bad[section][key]
        with pytest.raises(ConfigError):
            resolve_process_config(bad)
    raw["datasets"] = []
    with pytest.raises(ConfigError):
        resolve_process_config(raw)


def test_world_size_and_dtype_constraints(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ConfigError, match="one process"):
        resolve_process_config(minimal())
    monkeypatch.setenv("WORLD_SIZE", "1")
    raw = minimal()
    raw["model"]["dtype"] = "fp32"
    with pytest.raises(ConfigError, match="agree"):
        resolve_process_config(raw)
    raw["train"] = {"dtype": "float32", "batch_size": 2, "gradient_accumulation_steps": 2}
    cfg = resolve_process_config(raw)
    assert cfg["_gen2_resolved"]["effective_batch_size"] == 4
    assert cfg["model"]["dtype"] == "float32"


def test_tracked_examples_have_exact_user_prompt_and_vm_paths():
    root = Path(__file__).resolve().parents[1]
    for kind, steps, every in (("smoke", 4, 2), ("pilot", 500, 100)):
        path = root / "config" / f"{kind}_gen2_ideogram4_v2.example.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["config"]["process"][0]
        cfg = resolve_process_config(raw)
        assert cfg["train"]["steps"] == steps
        assert cfg["sample"]["sample_every"] == every
        assert cfg["save"]["save_every"] == every
        assert cfg["model"]["name_or_path"] == "ideogram-ai/ideogram-4-fp8"
        assert cfg["datasets"][0]["folder_path"] == "/data/train/datasets/en_ig4_r1X1dOn9mA2"
        prompt = cfg["sample"]["prompts"][0]
        assert hashlib.sha256(prompt.encode()).hexdigest() == "b57f5ce3d890147803624284a2a3ff6b337dd1107033aedcc132b297eb5de17f"
        assert prompt.count("[trigger]") == 3
        assert cfg["gen2"]["evaluation"]["named_phrase"] == "Ghibli anime"
