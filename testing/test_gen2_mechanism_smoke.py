import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

from extensions_built_in.gen2_trainer.activator import PlaceholderContract, SoftTokenBank, replace_token_spans_with_soft_tokens
from extensions_built_in.gen2_trainer.checkpoint import load_phase_checkpoint, save_phase_checkpoint
from extensions_built_in.gen2_trainer.config import validate_gen2_config
from extensions_built_in.gen2_trainer.registry import AdapterRuntimeContext, Gen2AdapterBank, TargetSpec, enable_adapter_training, install_ideogram_adapters
from extensions_built_in.gen2_trainer.temporal_rank_field import TemporalRankFieldLoRA
from extensions_built_in.gen2_trainer.sampling import build_validation_matrix, make_flowmatch_noisy_latents, sample_stratified_timesteps


ROOT = Path(__file__).parents[1]
SMOKE_CONFIG = ROOT / "config" / "gen2_trainer_ideogram4_smoke.yaml"


def _load_ideogram_modules():
    package = types.ModuleType("gen2_smoke_ideogram")
    package.__path__ = []
    sys.modules[package.__name__] = package
    transformer_name = "gen2_smoke_ideogram.transformer"
    transformer_spec = importlib.util.spec_from_file_location(
        transformer_name,
        ROOT / "extensions_built_in/diffusion_models/ideogram4/src/transformer.py",
    )
    transformer = importlib.util.module_from_spec(transformer_spec)
    sys.modules[transformer_name] = transformer
    transformer_spec.loader.exec_module(transformer)
    pipeline_name = "gen2_smoke_ideogram.pipeline"
    pipeline_spec = importlib.util.spec_from_file_location(
        pipeline_name,
        ROOT / "extensions_built_in/diffusion_models/ideogram4/src/pipeline.py",
    )
    pipeline = importlib.util.module_from_spec(pipeline_spec)
    sys.modules[pipeline_name] = pipeline
    pipeline_spec.loader.exec_module(pipeline)
    return pipeline, transformer


try:
    _PIPELINE, _TRANSFORMER = _load_ideogram_modules()
    predict_velocity = _PIPELINE.predict_velocity
    Ideogram4Config = _TRANSFORMER.Ideogram4Config
    Ideogram4Transformer2DModel = _TRANSFORMER.Ideogram4Transformer2DModel
    IDEOGRAM_RUNTIME_IMPORT_ERROR = None
except (ImportError, OSError) as error:
    _PIPELINE = None
    _TRANSFORMER = None
    IDEOGRAM_RUNTIME_IMPORT_ERROR = error


class MechanismSmokeTest(unittest.TestCase):
    def test_smoke_config_enables_every_mechanism(self):
        import yaml

        payload = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
        process = payload["config"]["process"][0]
        self.assertEqual(process["type"], "gen2_trainer")
        self.assertTrue(process["model"]["unconditional"]["enabled"])
        self.assertTrue(process["network"]["image_token_only"])
        self.assertGreater(len(process["datasets"][0]["helpers"]), 0)
        self.assertEqual(len({helper["id"] for helper in process["datasets"][0]["helpers"]}), len(process["datasets"][0]["helpers"]))
        cases = build_validation_matrix(1, [1], [0.0, 0.5, 1.0])
        self.assertEqual(
            [(case.conditioning_mode, case.style_gate, case.eta_c, case.eta_u) for case in cases],
            [
                ("native_helper", 0.0, 0.0, 0.0),
                ("soft_tokens", 0.0, 0.0, 0.0),
                ("native_helper", 1.0, 1.0, 0.0),
                ("soft_tokens", 1.0, 1.0, 0.0),
                ("soft_tokens", 1.0, 1.0, 0.5),
                ("soft_tokens", 1.0, 1.0, 1.0),
            ],
        )
        self.assertTrue(process["activator"]["trigger_local_adapter"]["enabled"])
        self.assertEqual(process["activator"]["initialization"]["strategy"], "literal_trigger_resampled")
        self.assertEqual(process["activator"]["tokens"], 24)
        self.assertTrue(process["train"]["phase_a"]["calibration"]["enabled"])
        self.assertIn("effect_geometry", process["train"]["phase_a"])
        self.assertGreaterEqual(process["train"]["phase_a"]["effect_geometry"]["cone_iterations"], 1)
        self.assertEqual(process["train"]["phase_a"]["curriculum"]["effective_batch_size"], 4)
        self.assertEqual(process["train"]["phase_a"]["batch_size"], 1)
        self.assertEqual(process["train"]["phase_b"]["batch_size"], 2)
        self.assertNotEqual(process["train"]["phase_a"]["batch_size"], process["train"]["phase_b"]["batch_size"])
        phase_a_save = process["save"]["phase_a"]
        phase_b_save = process["save"]["phase_b"]
        self.assertGreaterEqual(phase_a_save["save_every"], 0)
        self.assertGreaterEqual(phase_b_save["save_every"], 0)
        self.assertGreaterEqual(phase_a_save["max_step_saves_to_keep"], 0)
        self.assertGreaterEqual(phase_b_save["max_step_saves_to_keep"], 0)
        self.assertIsNot(phase_a_save, phase_b_save)
        self.assertTrue(process["sample"]["require_official_unconditional"])
        validate_gen2_config(process)

    def test_timestep_sampler_uses_model_scale_and_all_bins(self):
        torch.manual_seed(0)
        order = torch.arange(4)
        cursor = 0
        chunks = []
        for _ in range(2):
            timesteps, order, cursor = sample_stratified_timesteps(2, 4, order, cursor, torch.device("cpu"))
            chunks.append(timesteps)
        timesteps = torch.cat(chunks)
        self.assertTrue(torch.all((timesteps >= 0.0) & (timesteps <= 1000.0)))
        self.assertEqual(torch.unique(torch.floor(timesteps / 250.0).clamp_max(3)).numel(), 4)
        clean = torch.ones(4, 1, 1, 1)
        noise = torch.zeros_like(clean)
        mixed = make_flowmatch_noisy_latents(clean, noise, torch.tensor([0.0, 250.0, 500.0, 1000.0]))
        self.assertTrue(torch.allclose(mixed.flatten(), torch.tensor([1.0, 0.75, 0.5, 0.0])))

    def test_adapter_training_is_reenabled_after_parent_freeze(self):
        bank = Gen2AdapterBank()
        base = nn.Linear(4, 4)
        adapter = TemporalRankFieldLoRA(4, 4, rank=2)
        bank.add("linear", adapter, TargetSpec("linear", base, None))
        base.requires_grad_(False)
        bank.requires_grad_(False)
        self.assertFalse(any(parameter.requires_grad for parameter in bank.parameters()))
        trainable = enable_adapter_training(bank)
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.requires_grad for parameter in trainable))
        self.assertTrue(all(parameter.requires_grad for parameter in adapter.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in base.parameters()))

    def test_packed_velocity_and_adapter_gradient(self):
        if IDEOGRAM_RUNTIME_IMPORT_ERROR is not None:
            self.skipTest(f"Ideogram runtime dependencies unavailable: {IDEOGRAM_RUNTIME_IMPORT_ERROR}")
        config = Ideogram4Config(
            emb_dim=32,
            num_heads=4,
            in_channels=8,
            llm_dim=64,
            intermediate_size=64,
            num_layers=1,
        )
        transformer = Ideogram4Transformer2DModel(config)
        bank = install_ideogram_adapters(transformer, rank=2, alpha=2, knots=3, delta_max=1.0)
        for module in bank.adapters.values():
            for parameter in module.parameters():
                parameter.data.normal_(0, 0.1)
        latents = torch.randn(2, 8, 2, 2)
        features = torch.randn(2, 3, 64)
        mask = torch.ones(2, 3, dtype=torch.long)
        context = AdapterRuntimeContext(torch.tensor([0.2, 0.8]), torch.ones(2), 1.0)
        transformer.gradient_checkpointing = True
        output = predict_velocity(transformer, latents, torch.tensor([0.2, 0.8]), features, mask, adapter_context=context)
        self.assertEqual(output.shape, latents.shape)
        output.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in bank.parameters()))

    def test_trigger_local_adapter_supports_bfloat16_hidden_states(self):
        from extensions_built_in.gen2_trainer.activator import TriggerLocalTEAdapter

        module = TriggerLocalTEAdapter(8, rank=2, alpha=4).to(dtype=torch.float32)
        hidden = torch.randn(1, 4, 8, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
        output = module(hidden, mask)
        self.assertEqual(output.dtype, hidden.dtype)
        output.square().mean().backward()
        self.assertIsNotNone(module.up.weight.grad)
        self.assertIsNotNone(module.down.weight.grad)

    def test_image_only_mask_and_gate(self):
        module = TemporalRankFieldLoRA(4, 4, rank=2)
        module.up.data.fill_(1.0)
        x = torch.randn(1, 5, 4)
        mask = torch.tensor([[[1.0], [1.0], [0.0], [0.0], [0.0]]])
        output = module(x, torch.tensor([0.5]), mask, torch.ones(1), 1.0)
        self.assertTrue(torch.equal(output[:, 2:], torch.zeros_like(output[:, 2:])))
        self.assertTrue(torch.equal(module(x, torch.tensor([0.5]), mask, torch.zeros(1), 1.0), torch.zeros_like(output)))

    def test_checkpoint_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint"
            module = nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
            loss = module(torch.ones(2, 3)).square().mean()
            loss.backward()
            optimizer.step()
            save_phase_checkpoint(path, {"weight": module.weight.detach().clone()}, {"phase": "a", "step": 1}, optimizer)
            tensors, metadata = load_phase_checkpoint(path, optimizer)
        self.assertEqual(metadata["phase"], "a")
        self.assertTrue(torch.equal(tensors["weight"], module.weight.detach()))

    def test_placeholder_expansion_contract(self):
        contract = PlaceholderContract()
        raw = json.dumps({"a": "[trigger]", "b": ["x [trigger] y"]})
        _, occurrences = contract.parse(raw)
        self.assertEqual(len(occurrences), 2)
        bank = SoftTokenBank(3, 4)
        embeddings = torch.randn(10, 4)
        expanded, spans = replace_token_spans_with_soft_tokens(embeddings, [(1, 2), (5, 6)], bank)
        self.assertEqual(expanded.shape[0], 10 - 2 + 2 * 3)
        self.assertEqual(len(spans), 2)


if __name__ == "__main__":
    unittest.main()
