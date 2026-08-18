import ast
import contextlib
import importlib
import inspect
import os
import tempfile
import types
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch


def _load_runtime_methods():
    source_path = Path(__file__).parents[1] / 'extensions_built_in' / 'sd_trainer' / 'SDTrainer.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SDTrainer')
    names = {
        'three_phase_enabled', '_load_trigger_binding_modules', '_call_supported', '_first_callable',
        '_phase_config', '_phase_runtime_objective', '_activator_component_flags', '_configure_phase_trainability',
        'hook_add_extra_train_params', '_activator_mode', '_write_trigger_binding_metrics',
        '_phase_caption_source_weights', '_prompt_tap_batch', '_check_first_trigger_gradient', '_v8_masked_delta_metrics', '_v8_prompt_delta_norms',
        '_calculate_trigger_binding_loss', '_calculate_ideogram4_v3_activator_loss',
        '_v3_diagnostics_config', '_validate_frozen_v3_checkpoint_scope', '_v3_diagnostic_event',
        '_v3_segmented_smoothstep_weight', '_v3_schedule_weights',
        '_v3_probe_target_mse', '_evaluate_ideogram4_v3_fixed_probe',
        '_v8_validation_config', '_maybe_run_v8_fixed_validation',
        '_install_trigger_binding_prompt_encoder', 'encode_static_prompt',
        '_sync_semantic_scaffold_tap_runtime', 'end_step_hook',
    }
    selected = [node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    module = ast.Module(body=[ast.ClassDef(name='SDTrainerRuntimeHarness', bases=[ast.Name(id='RuntimeBase', ctx=ast.Load())], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(module)
    class RuntimeBase:
        def end_step_hook(self):
            pass
    from toolkit.semantic_scaffold_calibration import FixedDiffusionProbeCase

    @contextlib.contextmanager
    def network_disabled(network):
        previous = None if network is None else network.is_active
        if network is not None:
            network.is_active = False
        try:
            yield
        finally:
            if network is not None:
                network.is_active = previous

    namespace = {
        'RuntimeBase': RuntimeBase,
        'contextlib': contextlib,
        'importlib': importlib,
        'inspect': inspect,
        'os': os,
        'json': __import__('json'),
        'time': __import__('time'),
        'print_acc': lambda *_args, **_kwargs: None,
        'MethodType': MethodType,
        'Mapping': dict,
        'torch': torch,
        'F': torch.nn.functional,
        'get_torch_dtype': lambda _dtype: torch.float32,
        'shared_loss_target': lambda trainer, noise, batch, timesteps: trainer.sd.get_loss_target(
            noise=noise, batch=batch, timesteps=timesteps
        ).detach(),
        'should_run_validation': lambda completed, config: completed in tuple(config.steps),
        'write_fixed_probe_results': lambda *args, **kwargs: None,
        'FixedDiffusionProbeCase': FixedDiffusionProbeCase,
        'load_file': __import__('safetensors.torch', fromlist=['load_file']).load_file,
        'network_disabled': network_disabled,
    }
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['SDTrainerRuntimeHarness']


SDTrainer = _load_runtime_methods()


class _FakeEmbeds:
    def __init__(self, batch_size, token_count=3, active=True, with_taps=False):
        scale = 1.0 if active else 0.0
        self.text_embeds = [torch.full((token_count, 2), scale) for _ in range(batch_size)]
        self.trigger_masks = [
            torch.tensor([True] + [False] * (token_count - 1)) for _ in range(batch_size)
        ]
        if with_taps:
            taps = torch.zeros(13, token_count, 2)
            taps[:, 0, 0] = scale
            self.text_taps = [taps.clone() for _ in range(batch_size)]

    def __contains__(self, key):
        return hasattr(self, key)

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        detached = _FakeEmbeds.__new__(_FakeEmbeds)
        detached.text_embeds = [tensor.detach() for tensor in self.text_embeds]
        detached.trigger_masks = [tensor.detach() for tensor in self.trigger_masks]
        if hasattr(self, 'text_taps'):
            detached.text_taps = [tensor.detach() for tensor in self.text_taps]
        return detached


class _FakeActivator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(2, 2, bias=False)
        self.te_adapter = torch.nn.Linear(2, 2, bias=False)
        self.tap_adapters = torch.nn.ModuleDict({'0': torch.nn.Linear(2, 2, bias=False)})
        self.module_lora_adapters = torch.nn.ModuleDict()
        self.component_active = {}

    def set_component_mode(self, component, active=None, trainable=None):
        self.component_active[component] = active
        getattr(self, component).requires_grad_(trainable)

    def parameter_groups(self, learning_rates=None):
        learning_rates = learning_rates or {}
        groups = []
        for name in ('embedding', 'te_adapter', 'tap_adapters'):
            params = [parameter for parameter in getattr(self, name).parameters() if parameter.requires_grad]
            if params:
                group = {'params': params, 'name': name}
                if name in learning_rates:
                    group['lr'] = learning_rates[name]
                groups.append(group)
        return groups


class ThreePhaseRuntimeTest(unittest.TestCase):
    def _trainer(self, phase):
        trainer = SDTrainer.__new__(SDTrainer)
        trainer.runtime_phase = phase
        trainer.text_activator = _FakeActivator()
        trainer.network = torch.nn.Linear(2, 2, bias=False)
        phase_config = SimpleNamespace(
            train={
                'embedding': phase != 'b',
                'internal': False,
                'tap': phase == 'a2',
                'diffusion_lora': phase == 'b',
            },
            learning_rates={'embedding': 1e-3, 'tap_adapters': 2e-3},
            losses={'diffusion_mse': {'enabled': True, 'weight': 1.0}},
            activator_gain_floor=SimpleNamespace(
                enabled=False,
                weight=0.0,
                schedule=SimpleNamespace(keyframes=[], interpolation='smoothstep'),
            ),
            context_consistency=SimpleNamespace(
                enabled=False,
                weight=0.0,
                pooling='mean',
                detach_reference=False,
                magnitude_weight=0.0,
                min_delta_norm=1e-6,
                warmup_steps=0,
                tap_layers=None,
            ),
        )
        trainer.three_phase_trigger_training = SimpleNamespace(
            enabled=True,
            phase_a1=phase_config,
            phase_b=phase_config,
            phase_a2=phase_config,
            phase_runtime=SimpleNamespace(caption_sources={}, objective={}),
            run_root=None,
        )
        trainer._v3_diagnostic_sequence = 0
        return trainer

    def test_end_step_hook_runs_validation_after_completed_update_increment(self):
        trainer = self._trainer('a1')
        observed_steps = []
        trainer.step_num = 8
        trainer._maybe_run_v8_fixed_validation = lambda: observed_steps.append(trainer.step_num)

        trainer.end_step_hook()

        self.assertEqual(observed_steps, [8])

    def test_v3_validation_uses_incremented_completed_update_without_plus_one(self):
        trainer = self._trainer('a1')
        trainer.step_num = 4
        trainer.save_root = tempfile.gettempdir()
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer._trigger_binding_probe_sets = {'train': [{}], 'heldout': [{}]}
        trainer._trigger_binding_validation_steps = set()
        trainer.three_phase_trigger_training.validation = SimpleNamespace(
            enabled=True,
            steps=[5],
        )
        trainer.three_phase_trigger_training.phase_runtime.objective = {
            'pipeline': 'ideogram4_v3_activator',
            'diagnostics': {'enabled': False},
        }
        trainer._evaluate_ideogram4_v3_fixed_probe = lambda item, split, step: {}
        trainer.trigger_word = None
        trainer.semantic_scaffold_enabled = False
        trainer.ideogram4_v3_activator_enabled = True

        trainer._maybe_run_v8_fixed_validation()
        self.assertEqual(trainer._trigger_binding_validation_steps, set())

        trainer.step_num = 5
        trainer._maybe_run_v8_fixed_validation()
        self.assertEqual(trainer._trigger_binding_validation_steps, {5})

    def test_phase_whitelist_freezes_non_targets(self):
        trainer = self._trainer('a2')
        params = [{'params': list(trainer.network.parameters())}]
        filtered = trainer.hook_add_extra_train_params(params)
        selected = {id(parameter) for group in filtered for parameter in group['params']}
        self.assertTrue(all(id(parameter) not in selected for parameter in trainer.network.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in trainer.text_activator.embedding.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in trainer.text_activator.tap_adapters.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.text_activator.te_adapter.parameters()))

    def test_b_phase_keeps_only_diffusion_lora_and_freezes_activator(self):
        trainer = self._trainer('b')
        params = [{'params': list(trainer.network.parameters()) + list(trainer.text_activator.parameters())}]
        filtered = trainer.hook_add_extra_train_params(params)
        selected = {id(parameter) for group in filtered for parameter in group['params']}
        self.assertEqual(selected, {id(parameter) for parameter in trainer.network.parameters()})
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.text_activator.parameters()))

    def test_v3_builder_tokenizes_before_registration_and_disables_taps(self):
        source_path = Path(__file__).parents[1] / 'extensions_built_in' / 'sd_trainer' / 'SDTrainer.py'
        source = source_path.read_text(encoding='utf-8')
        pre_registration = source.index("source_ids = list(tokenizer(literal, add_special_tokens=False)")
        registration = source.index("tokenizer.add_tokens([literal], special_tokens=True)")
        self.assertLess(pre_registration, registration)
        self.assertIn(
            "create_tap_adapters=bool(tap_config.enabled and not is_v3_activator)",
            source,
        )
        self.assertIn(
            "extensions_built_in.ideogram4_v3_activator.literal_initialization",
            source,
        )

    def test_v3_segmented_schedule_boundaries_are_exact(self):
        trainer = self._trainer('a1')
        self.assertEqual(trainer._v3_schedule_weights(0), {'helper': 1.0, 'dataset': 0.25})
        self.assertEqual(trainer._v3_schedule_weights(40), {'helper': 0.1, 'dataset': 1.0})
        self.assertEqual(trainer._v3_schedule_weights(80), {'helper': 0.0, 'dataset': 1.0})

    def test_v3_te_calibration_freezes_embedding_taps_and_active_network(self):
        trainer = self._trainer('a2')
        trainer.three_phase_trigger_training.phase_runtime.objective = {
            'pipeline': 'ideogram4_v3_activator',
            'name': 'te_calibration',
            'te_only': True,
            'v3_active_frozen': True,
        }
        params = [{'params': list(trainer.network.parameters()) + list(trainer.text_activator.parameters())}]
        filtered = trainer.hook_add_extra_train_params(params)
        selected = {id(parameter) for group in filtered for parameter in group['params']}
        self.assertEqual(selected, {id(parameter) for parameter in trainer.text_activator.te_adapter.parameters()})
        self.assertTrue(trainer.network.is_active)
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.network.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.text_activator.embedding.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.text_activator.tap_adapters.parameters()))

    def test_a2_rejects_combined_te_and_diffusion_v3_checkpoint(self):
        from safetensors.torch import save_file

        trainer = self._trainer('a2')
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'combined.safetensors')
            save_file({
                'lora_unet$$layers$$0.lora_down.weight': torch.ones(1, 1),
                'lora_te$$language_model$$layers$$0.lora_down.weight': torch.ones(1, 1),
            }, path)
            trainer.three_phase_trigger_training.phase_runtime.objective = {
                'pipeline': 'ideogram4_v3_activator',
                'v3_weights_path': path,
                'diagnostics': {'require_diffusion_only_v3': True},
            }
            with self.assertRaisesRegex(RuntimeError, 'diffusion-only frozen V3 LoRA'):
                trainer._validate_frozen_v3_checkpoint_scope()

    def test_a2_accepts_diffusion_only_v3_checkpoint(self):
        from safetensors.torch import save_file

        trainer = self._trainer('a2')
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'diffusion.safetensors')
            save_file({
                'lora_unet$$layers$$0.lora_down.weight': torch.ones(1, 1),
                'lora_unet$$layers$$0.lora_up.weight': torch.ones(1, 1),
            }, path)
            trainer.three_phase_trigger_training.phase_runtime.objective = {
                'pipeline': 'ideogram4_v3_activator',
                'v3_weights_path': path,
                'diagnostics': {'require_diffusion_only_v3': True},
            }
            trainer._validate_frozen_v3_checkpoint_scope()

    def test_v3_runtime_diagnostics_persist_ordered_events(self):
        trainer = self._trainer('a1')
        trainer.step_num = 2
        trainer.device_torch = torch.device('cpu')
        trainer._v3_diagnostic_sequence = 0
        with tempfile.TemporaryDirectory() as root:
            trainer.save_root = root
            trainer.three_phase_trigger_training.run_root = root
            trainer.three_phase_trigger_training.phase_runtime.objective = {
                'pipeline': 'ideogram4_v3_activator',
                'name': 'semantic_activator',
                'diagnostics': {
                    'enabled': True,
                    'filename': 'runtime_diagnostics.jsonl',
                    'console': False,
                    'fsync': True,
                },
            }
            trainer._v3_diagnostic_event('diffusion_forward_start', label='helper:one')
            trainer._v3_diagnostic_event('diffusion_forward_end', label='helper:one')
            path = os.path.join(root, 'phase_a1', 'runtime_diagnostics.jsonl')
            with open(path, 'r', encoding='utf-8') as handle:
                records = [__import__('json').loads(line) for line in handle if line.strip()]
        self.assertEqual([record['sequence'] for record in records], [1, 2])
        self.assertEqual([record['event'] for record in records], [
            'diffusion_forward_start', 'diffusion_forward_end'
        ])
        self.assertEqual(records[0]['step_index'], 2)
        self.assertEqual(records[0]['label'], 'helper:one')

    def test_v3_a1_detached_teacher_predictions_run_without_autograd(self):
        from toolkit import trigger_binding_losses

        trainer = self._trainer('a1')
        trainer.device_torch = torch.device('cpu')
        trainer.step_num = 0
        trainer.additional_logs = {}
        trainer._trigger_gradient_reachability_checked = True
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.phase_runtime.objective = {
            'pipeline': 'ideogram4_v3_activator',
            'name': 'semantic_activator',
            'helper_schedule': {'helpers': ['one', 'two']},
        }
        trainer._trigger_binding_modules = {'losses': trigger_binding_losses}
        trainer._check_first_trigger_gradient = lambda *_args, **_kwargs: self.fail(
            'V3 objective must not run an extra retained-graph reachability backward'
        )
        trainer.sd = SimpleNamespace(
            get_prompt_embeds=lambda prompts: _FakeEmbeds(len(prompts)),
            get_loss_target=lambda noise, batch, timesteps: noise,
        )
        trainer._activator_mode = lambda mode: patch.object(trainer, '_mode', mode, create=True)
        grad_states = []

        def predict_noise(**kwargs):
            grad_states.append((trainer._mode, torch.is_grad_enabled()))
            scale = 1.0 if trainer._mode == 'semantic_only' else 0.5
            return kwargs['noisy_latents'] * scale

        trainer.predict_noise = predict_noise
        latents = torch.ones(1, 4, 2, 3, requires_grad=True)
        batch = SimpleNamespace(
            file_items=[SimpleNamespace(caption_template='x [trigger]', raw_caption='unused')],
        )
        trainer._calculate_ideogram4_v3_activator_loss(
            noisy_latents=latents,
            noise=torch.zeros_like(latents),
            timesteps=torch.tensor([10]),
            batch=batch,
            pred_kwargs={},
            dtype=torch.float32,
        )
        self.assertEqual(grad_states[0], ('semantic_only', True))
        self.assertTrue(all(not enabled for _mode, enabled in grad_states[1:]))

    def test_v3_a1_prototype_is_stable_across_bucket_shapes(self):
        from toolkit import trigger_binding_losses

        trainer = self._trainer('a1')
        trainer.device_torch = torch.device('cpu')
        trainer.step_num = 0
        trainer.additional_logs = {}
        trainer._trigger_gradient_reachability_checked = True
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.phase_runtime.objective = {
            'pipeline': 'ideogram4_v3_activator',
            'name': 'semantic_activator',
            'helper_schedule': {'helpers': ['illustration']},
            'prototype_ema_decay': 0.95,
        }
        trainer._trigger_binding_modules = {'losses': trigger_binding_losses}
        trainer.sd = SimpleNamespace(
            get_prompt_embeds=lambda prompts: _FakeEmbeds(len(prompts)),
            get_loss_target=lambda noise, batch, timesteps: noise,
        )
        trainer._activator_mode = lambda mode: patch.object(trainer, '_mode', mode, create=True)
        batch = SimpleNamespace(
            file_items=[SimpleNamespace(caption_template='x [trigger]', raw_caption='unused')],
        )

        def run_shape(height, width):
            def predict_noise(**kwargs):
                if trainer._mode == 'semantic_only':
                    return torch.ones_like(kwargs['noisy_latents'])
                if trainer._mode == 'stock_literal':
                    return torch.full_like(kwargs['noisy_latents'], 0.5)
                return torch.zeros_like(kwargs['noisy_latents'])
            trainer.predict_noise = predict_noise
            latents = torch.zeros(1, 4, height, width)
            return trainer._calculate_ideogram4_v3_activator_loss(
                noisy_latents=latents,
                noise=torch.zeros_like(latents),
                timesteps=torch.tensor([10]),
                batch=batch,
                pred_kwargs={},
                dtype=torch.float32,
            )

        self.assertTrue(torch.isfinite(run_shape(4, 5)))
        self.assertEqual(tuple(trainer._v3_semantic_prototype.shape), (4,))
        self.assertTrue(torch.isfinite(run_shape(3, 7)))
        self.assertEqual(tuple(trainer._v3_semantic_prototype.shape), (4,))

    def test_v3_a2_objective_has_one_prediction_and_no_helper(self):
        from toolkit import trigger_binding_losses

        trainer = self._trainer('a2')
        trainer.device_torch = torch.device('cpu')
        trainer.step_num = 0
        trainer.additional_logs = {}
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.phase_runtime.objective = {
            'pipeline': 'ideogram4_v3_activator',
            'name': 'te_calibration',
            'helper_schedule': {'helpers': ['illustration']},
        }
        trainer._trigger_binding_modules = {'losses': trigger_binding_losses}
        trainer.sd = SimpleNamespace(
            get_prompt_embeds=lambda prompts: _FakeEmbeds(len(prompts)),
            get_loss_target=lambda noise, batch, timesteps: noise + 1,
        )
        calls = []
        trainer.predict_noise = lambda **kwargs: calls.append(kwargs) or kwargs['noisy_latents'] * 0.0
        trainer._activator_mode = lambda mode: patch.object(trainer, '_mode', mode, create=True)
        batch = SimpleNamespace(
            file_items=[SimpleNamespace(caption_template='x [trigger]', raw_caption='unused')],
        )
        loss = trainer._calculate_ideogram4_v3_activator_loss(
            noisy_latents=torch.zeros(1, 2),
            noise=torch.zeros(1, 2),
            timesteps=torch.tensor([10]),
            batch=batch,
            pred_kwargs={},
            dtype=torch.float32,
        )
        self.assertEqual(len(calls), 1)
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(trainer._trigger_binding_last_metrics['v3/a2/helper_passes'], 0.0)

    def test_v3_fixed_probe_runs_diffusion_conditions_and_restores_network(self):
        trainer = self._trainer('a2')
        trainer.device_torch = torch.device('cpu')
        trainer.step_num = 0
        trainer.additional_logs = {}
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.literal = '<atomic>'
        trainer.three_phase_trigger_training.phase_runtime.objective = {
            'pipeline': 'ideogram4_v3_activator',
            'helper_schedule': {'helpers': ['helper-one', 'helper-two']},
        }
        trainer.text_activator.original_literal_text = 'old literal words'
        trainer.network.is_active = False
        prepared = {
            'latent': torch.zeros(1, 2),
            'noise': torch.ones(1, 2),
            'noisy_latents': torch.zeros(1, 2),
            'timesteps': torch.tensor([250.0]),
            'target': torch.ones(1, 2),
            'prompt_template': 'render [trigger] object',
            'placeholder': '[trigger]',
        }
        trainer._semantic_scaffold_prepare_case = lambda case: prepared
        trainer._semantic_scaffold_probe_batch = lambda value: SimpleNamespace(latents=value['latent'])
        calls = []

        class _Embeds:
            def __init__(self, label):
                self.label = label

            def to(self, *_args, **_kwargs):
                return self

        trainer.sd = SimpleNamespace(get_prompt_embeds=lambda prompts: _Embeds(('bound', trainer._mode, prompts[0])))
        trainer._trigger_binding_prompt_encoder = lambda prompts, runtime_mode: _Embeds(('text', runtime_mode, prompts[0]))
        trainer._activator_mode = lambda mode: patch.object(trainer, '_mode', mode, create=True)

        def predict_noise(**kwargs):
            label = kwargs['conditional_embeds'].label
            calls.append((label, trainer.network.is_active))
            if label[0] == 'bound' and not trainer.network.is_active:
                return torch.zeros(1, 2)
            if label[0] == 'text' and 'old literal words' in label[2]:
                return torch.full((1, 2), 0.25)
            if label[0] == 'bound':
                return torch.full((1, 2), 0.75)
            return torch.full((1, 2), 0.5)

        trainer.predict_noise = predict_noise
        case = {
            'probe_case_id': 'fixed-case', 'split': 'heldout', 'item_id': 'image.png',
            'caption_hash': 'caption', 'image_hash': 'image', 'noise_seed': 42,
            'timestep': 250, 'sigma': None, 'target_mode': 'flow', 'transform': {},
            'dataset_relative_item_id': 'image.png', 'probe_index': 0,
        }
        result = trainer._evaluate_ideogram4_v3_fixed_probe(case, 'heldout', 0)

        self.assertFalse(trainer.network.is_active)
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[0][0][:2], ('bound', 'semantic_only'))
        self.assertFalse(calls[0][1])
        self.assertEqual(calls[1][0], ('text', 'stock_literal', 'render old literal words object'))
        self.assertTrue(calls[1][1])
        self.assertEqual(calls[2][0][:2], ('bound', 'semantic_only'))
        self.assertTrue(calls[2][1])
        self.assertNotEqual(result['a_target_mse'], result['c_target_mse'])
        self.assertAlmostEqual(result['heldout_loss'], 0.0625)
        self.assertEqual(result['heldout_loss'], result['c_target_mse'])
        self.assertEqual(set(result['conditions']['D_v3_helpers']), {'helper-one', 'helper-two'})

    def test_semantic_scaffold_handoff_freezes_semantic_groups_without_forward_toggle(self):
        trainer = self._trainer('a1')
        trainer.semantic_scaffold_enabled = True
        trainer._semantic_scaffold_state = SimpleNamespace(
            curriculum=SimpleNamespace(unlock_step=2)
        )
        trainer._semantic_scaffold_tap_lr_ramp = 0.5
        trainer.three_phase_trigger_training.phase_a1.learning_rates.update({
            'te_adapter': 3e-3,
        })
        trainer._phase_config = lambda: trainer.three_phase_trigger_training.phase_a1
        groups = trainer.text_activator.parameter_groups(
            trainer.three_phase_trigger_training.phase_a1.learning_rates
        )
        for group in groups:
            group['name'] = f"text_activator.{group['name']}"
        trainer.params = groups
        trainer.optimizer = None

        trainer._sync_semantic_scaffold_tap_runtime()

        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.text_activator.embedding.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.text_activator.te_adapter.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in trainer.text_activator.tap_adapters.parameters()))
        lrs = {group['name']: group['lr'] for group in groups}
        self.assertEqual(lrs['text_activator.embedding'], 0.0)
        self.assertEqual(lrs['text_activator.te_adapter'], 0.0)
        self.assertEqual(lrs['text_activator.tap_adapters'], 1e-3)

    def test_static_prompt_bypasses_required_trigger_binding(self):
        trainer = self._trainer('a1')
        modes = []

        class _ModeContext:
            def __init__(self, mode):
                self.mode = mode

            def __enter__(self):
                modes.append(self.mode)

            def __exit__(self, *_args):
                modes.append('restored')

        trainer._activator_mode = lambda mode: _ModeContext(mode)
        trainer.sd = SimpleNamespace(
            encode_prompt=lambda prompt, **_kwargs: ('encoded', prompt),
        )

        result = trainer.encode_static_prompt([''])

        self.assertEqual(result, ('encoded', ['']))
        self.assertEqual(modes, ['activator_bypass', 'restored'])

    def test_prompt_encoder_allows_static_prompt_only_in_bypass_mode(self):
        trainer = self._trainer('a1')
        original_calls = []

        class _SD:
            text_activator_runtime_mode = 'activator_bypass'

            def get_prompt_embeds(self, prompt, **kwargs):
                original_calls.append((prompt, kwargs))
                return ('plain', prompt)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        trainer._install_trigger_binding_prompt_encoder(SimpleNamespace())

        result = trainer.sd.get_prompt_embeds([''])

        self.assertEqual(result, ('plain', ['']))
        self.assertEqual(original_calls[0][1]['runtime_mode'], 'activator_bypass')

    def test_prompt_encoder_bypass_keeps_binding_metadata_for_trigger_caption(self):
        trainer = self._trainer('a1')

        class _Batch:
            input_ids = torch.tensor([[1, 2, 3]])
            attention_mask = torch.tensor([[1, 1, 1]])
            trigger_mask = torch.tensor([[False, True, False]])

            def to(self, _device):
                return self

        class _SD:
            text_activator_runtime_mode = 'activator_bypass'
            tokenizer = object()
            text_encoder = SimpleNamespace(device=torch.device('cpu'))
            torch_dtype = torch.float32
            max_text_length = 16

            def get_prompt_embeds(self, prompt, **kwargs):
                return ('plain', prompt, kwargs)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        modules = SimpleNamespace(bind_trigger_batch=lambda *_args, **_kwargs: _Batch())
        fake_pipeline = SimpleNamespace(get_qwen3_vl_features=lambda *_args, **_kwargs: (
            torch.ones(1, 3, 2), [torch.ones(1, 3, 2) for _ in range(13)]
        ))
        with patch('importlib.import_module', return_value=fake_pipeline):
            trainer._install_trigger_binding_prompt_encoder({'runtime': modules})
            embeds = trainer.sd.get_prompt_embeds(['x [trigger]'], return_taps=True)

        self.assertIn('text_taps', embeds)
        self.assertIn('trigger_masks', embeds)
        self.assertEqual(embeds.text_taps[0].shape, (13, 3, 2))
        torch.testing.assert_close(embeds.trigger_masks[0], torch.tensor([False, True, False]))

    def test_prompt_encoder_bypasses_already_injected_literal_caption(self):
        trainer = self._trainer('a1')
        original_calls = []

        class _SD:
            text_activator_runtime_mode = 'full'

            def get_prompt_embeds(self, prompt, **kwargs):
                original_calls.append((prompt, kwargs))
                return ('plain', prompt)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        trainer._install_trigger_binding_prompt_encoder(SimpleNamespace())

        result = trainer.sd.get_prompt_embeds(['caption with <trigger> already injected'])

        self.assertEqual(result, ('plain', ['caption with <trigger> already injected']))
        self.assertEqual(original_calls[0][1]['runtime_mode'], 'activator_bypass')

    def test_prompt_encoder_forwards_return_taps_in_bypass_without_breaking_calls(self):
        trainer = self._trainer('a1')
        original_calls = []

        class _SD:
            text_activator_runtime_mode = 'activator_bypass'

            def get_prompt_embeds(self, prompt, **kwargs):
                original_calls.append(kwargs)
                return ('plain', prompt)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        trainer._install_trigger_binding_prompt_encoder(SimpleNamespace())

        trainer.sd.get_prompt_embeds([''], return_taps=True)

        self.assertTrue(original_calls[0]['return_taps'])
        self.assertEqual(original_calls[0]['runtime_mode'], 'activator_bypass')

    def test_prompt_encoder_still_rejects_caption_without_placeholder_or_literal(self):
        trainer = self._trainer('a1')

        class _SD:
            text_activator_runtime_mode = 'full'

            def get_prompt_embeds(self, prompt, **kwargs):
                return ('plain', prompt, kwargs)

        trainer.sd = _SD()
        trainer.three_phase_trigger_training.literal = '<trigger>'
        trainer.three_phase_trigger_training.placeholder = '[trigger]'
        trainer.three_phase_trigger_training.mask_all_occurrences = True
        trainer._install_trigger_binding_prompt_encoder(SimpleNamespace())

        with self.assertRaisesRegex(ValueError, 'every training caption must contain'):
            trainer.sd.get_prompt_embeds(['caption without the required token'])

    def test_phase_metrics_are_written_independently_and_once_per_step(self):
        trainer = self._trainer('a1')
        trainer.step_num = 7
        trainer._trigger_binding_last_metrics = {'gain': 0.25}
        trainer._trigger_binding_last_metrics_written_step = None
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.save_root = temp_dir
            trainer.three_phase_trigger_training.run_root = temp_dir
            trainer.three_phase_trigger_training.artifacts = SimpleNamespace(
                phase_a1=SimpleNamespace(metrics_file='metrics.jsonl'),
            )
            trainer._trigger_binding_metrics_pending_step = 7
            trainer._write_trigger_binding_metrics(torch.tensor(0.5))
            trainer._write_trigger_binding_metrics(torch.tensor(0.75))
            metrics_path = Path(temp_dir) / 'phase_a1' / 'metrics.jsonl'
            records = metrics_path.read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(records), 1)
            self.assertIn('"step": 7', records[0])
            self.assertIn('"loss": 0.5', records[0])
            self.assertIn('"gain": 0.25', records[0])

    def test_a_phase_single_source_fallback_shares_latent_noise_timestep_and_target(self):
        from toolkit import trigger_binding_losses

        trainer = self._trainer('a1')
        trainer.device_torch = torch.device('cpu')
        trainer.do_long_prompts = False
        trainer.additional_logs = {}
        trainer.sd = SimpleNamespace(
            get_prompt_embeds=lambda prompts, **kwargs: _FakeEmbeds(len(prompts)),
            get_loss_target=lambda noise, batch, timesteps: noise + 1,
        )
        calls = []

        def predict(noisy_latents, timesteps, **kwargs):
            calls.append((noisy_latents, timesteps, kwargs['conditional_embeds']))
            if getattr(trainer, '_mode', None) == 'activator_bypass':
                return noisy_latents.detach() + 1.0
            return noisy_latents * trainer.text_activator.embedding.weight.mean()

        trainer.predict_noise = predict
        trainer._activator_mode = lambda mode: patch.object(trainer, '_mode', mode, create=True)
        trainer._check_first_trigger_gradient = lambda *_args: None
        batch = SimpleNamespace(
            file_items=[SimpleNamespace(caption_template='x [trigger]', raw_caption='unused')],
            latents=torch.zeros(1, 2),
        )
        noisy = torch.randn(1, 2)
        noise = torch.randn(1, 2)
        timesteps = torch.tensor([10])
        trainer._trigger_binding_modules = {'losses': trigger_binding_losses}
        trainer._write_trigger_binding_metrics = lambda _loss: None
        loss = trainer._calculate_trigger_binding_loss(
            noisy, noise, timesteps, batch, {}, 1.0, torch.float32
        )
        self.assertTrue(loss.requires_grad)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[0] is noisy and call[1] is timesteps for call in calls))
        self.assertIn('a1/source/primary/diffusion_mse', trainer._trigger_binding_last_metrics)
        self.assertIn('a1/aggregate_loss', trainer._trigger_binding_last_metrics)

    def test_paired_sources_use_all_templates_and_add_context_once(self):
        from toolkit import trigger_binding_losses

        trainer = self._trainer('a2')
        trainer.device_torch = torch.device('cpu')
        trainer.do_long_prompts = False
        trainer.step_num = 0
        trainer.additional_logs = {}
        trainer.three_phase_trigger_training.phase_runtime.caption_sources = {
            'enabled': True,
            'sources': [{'name': 'structured'}, {'name': 'natural'}],
            'schedule': {
                'normalize_weights': True,
                'keyframes': [{'step': 0, 'structured': 0.75, 'natural': 0.25}],
            },
        }
        phase = trainer._phase_config()
        phase.context_consistency.enabled = True
        phase.context_consistency.weight = 0.5
        phase.context_consistency.tap_layers = list(range(13))
        encoded_prompts = []

        def encode(prompts, **kwargs):
            encoded_prompts.append((tuple(prompts), kwargs.get('return_taps')))
            token_count = 3 if prompts[0].startswith('S') else 5
            active = getattr(trainer, '_mode', None) == 'full'
            embeds = _FakeEmbeds(len(prompts), token_count=token_count, active=active, with_taps=True)
            if active and prompts[0].startswith('Natural'):
                for taps in embeds.text_taps:
                    taps[:, 0, 0] = 0.0
                    taps[:, 0, 1] = 1.0
            return embeds

        trainer.sd = SimpleNamespace(
            get_prompt_embeds=encode,
            get_loss_target=lambda noise, batch, timesteps: noise,
        )
        trainer.predict_noise = lambda noisy_latents, **kwargs: (
            noisy_latents * trainer.text_activator.embedding.weight.mean()
            if getattr(trainer, '_mode', None) == 'full' else noisy_latents.detach() + 1.0
        )
        trainer._activator_mode = lambda mode: patch.object(trainer, '_mode', mode, create=True)
        trainer._check_first_trigger_gradient = lambda *_args: None
        trainer._trigger_binding_modules = {'losses': trigger_binding_losses}
        trainer._write_trigger_binding_metrics = lambda _loss: None
        items = [SimpleNamespace(
            caption_template='fallback [trigger]',
            raw_caption='unused',
            caption_source_templates={'structured': 'S [trigger]', 'natural': 'Natural words [trigger]'},
        )]
        batch = SimpleNamespace(
            file_items=items,
            get_caption_source_templates=lambda names: [item.caption_source_templates[name] for item, name in zip(items, names)],
        )
        noisy = torch.ones(1, 2)
        loss = trainer._calculate_trigger_binding_loss(
            noisy, torch.zeros_like(noisy), torch.tensor([5]), batch, {}, 1.0, torch.float32
        )
        self.assertTrue(loss.requires_grad)
        self.assertEqual(len(encoded_prompts), 4)
        self.assertTrue(all(return_taps for _, return_taps in encoded_prompts))
        metrics = trainer._trigger_binding_last_metrics
        self.assertEqual(metrics['a2/source_weight/structured'], 0.75)
        self.assertEqual(metrics['a2/source_weight/natural'], 0.25)
        self.assertIn('a2/source/structured/activator_gain', metrics)
        self.assertIn('a2/source/natural/gain_floor_loss', metrics)
        self.assertGreater(metrics['a2/context_weighted'], 0.0)
        self.assertAlmostEqual(
            metrics['a2/aggregate_loss'],
            metrics['a2/aggregate_source_objective'] + metrics['a2/context_weighted'],
            places=6,
        )

    def test_first_real_loss_reachability_runs_once_and_raises(self):
        trainer = self._trainer('a1')
        trainer.params = list(trainer.text_activator.parameters())
        trainer.optimizer = torch.optim.SGD(trainer.params, lr=0.1)
        trainer._trigger_gradient_reachability_checked = False
        loss = trainer.text_activator.embedding.weight.square().mean()
        calls = []

        def check(*args, **kwargs):
            calls.append((args, kwargs))

        fake_module = SimpleNamespace(check_gradient_reachability=check)
        with patch('importlib.import_module', return_value=fake_module):
            trainer._check_first_trigger_gradient(loss, torch.ones(1), torch.zeros(1))
            trainer._check_first_trigger_gradient(loss, torch.ones(1), torch.zeros(1))
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1]['raise_on_error'])
        self.assertTrue(calls[0][1]['require_output_difference'])

        trainer._trigger_gradient_reachability_checked = False
        fake_module.check_gradient_reachability = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('unreachable'))
        with patch('importlib.import_module', return_value=fake_module):
            with self.assertRaisesRegex(RuntimeError, 'unreachable'):
                trainer._check_first_trigger_gradient(loss, torch.ones(1), torch.zeros(1))
        self.assertFalse(trainer._trigger_gradient_reachability_checked)

    def test_v8_zero_initialized_reachability_does_not_require_initial_output_difference(self):
        trainer = self._trainer('a1')
        trainer.three_phase_trigger_training.schema_version = 8
        trainer.three_phase_trigger_training.objective_mode = 'conditional_response_v8'
        trainer.params = list(trainer.text_activator.parameters())
        trainer.optimizer = torch.optim.SGD(trainer.params, lr=0.1)
        trainer._trigger_gradient_reachability_checked = False
        loss = trainer.text_activator.embedding.weight.square().mean()
        calls = []
        fake_module = SimpleNamespace(
            check_gradient_reachability=lambda *args, **kwargs: calls.append(kwargs)
        )
        with patch('importlib.import_module', return_value=fake_module):
            trainer._check_first_trigger_gradient(loss, torch.ones(1), torch.ones(1))
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]['require_output_difference'])
        self.assertTrue(calls[0]['raise_on_error'])

    def test_context_enabled_fails_fast_without_paired_sources(self):
        trainer = self._trainer('a2')
        trainer._phase_config().context_consistency.enabled = True
        trainer.three_phase_trigger_training.phase_runtime.caption_sources = {}
        trainer.sd = SimpleNamespace()
        batch = SimpleNamespace(file_items=[SimpleNamespace(caption_template='x [trigger]', raw_caption='x')])
        with self.assertRaisesRegex(ValueError, 'requires at least two'):
            trainer._calculate_trigger_binding_loss(
                torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([1]), batch, {}, 1.0, torch.float32
            )


if __name__ == '__main__':
    unittest.main()
