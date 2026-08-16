import unittest

from toolkit.config_modules import ThreePhaseTriggerTrainingConfig, validate_three_phase_trigger_training_config


class SemanticScaffoldConfigTest(unittest.TestCase):
    def _raw(self):
        return {
            'enabled': True,
            'schema_version': 8,
            'objective_mode': 'semantic_scaffold_control_channel',
            'execution': {'start_phase': 'a1', 'stop_after_phase': 'a1'},
            'trigger': {'placeholder': '[trigger]', 'literal': '<x>', 'span_detection': 'offsets', 'mask_all_occurrences': True, 'occurrence_mode': 'additive'},
            'text_activator': {
                'embedding': {'enabled': True, 'tokens': 4, 'init_mode': 'random'},
                'te_adapter': {'enabled': True, 'mode': 'module_lora', 'type': 'lora', 'rank': 4, 'alpha': 4, 'target_modules': ['down_proj'], 'token_mask_mode': 'trigger_span'},
                'tap_adapters': {'enabled': True, 'type': 'lora', 'rank': 2, 'alpha': 2, 'tap_layers': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35], 'token_mask_mode': 'trigger_span'},
            },
            'reachability_probe': {'enabled': True},
            'validation': {'enabled': False},
            'phase_a1': {
                'enabled': True, 'steps': 5, 'learning_rates': {'embedding': 0.1, 'te_adapter': 0.1, 'tap_adapters': 0.1},
                'train': {'embedding': True, 'te_adapter': True, 'tap_adapters': True},
                'losses': {'semantic_scaffold_control_channel': {'calibration': {'neutral_phrase': 'neutral', 'helper_candidates': ['helper'], 'noise_seeds': [1], 'fixed_timesteps': [10], 'max_helpers': 1}}},
            },
            'phase_b': {'enabled': False, 'steps': 0},
            'phase_a2': {'enabled': False, 'steps': 0},
        }

    def test_a1_only_semantic_objective_is_valid(self):
        config = ThreePhaseTriggerTrainingConfig(**self._raw())
        validate_three_phase_trigger_training_config(config, '<x>')
        self.assertEqual(config.execution_phase_names(), ('a1',))

    def test_semantic_objective_rejects_b_execution(self):
        raw = self._raw()
        raw['execution'] = {'start_phase': 'a1', 'stop_after_phase': 'b'}
        raw['phase_b'] = {'enabled': True, 'steps': 1, 'learning_rates': {'diffusion_lora': 0.1}, 'train': {'diffusion_lora': True}}
        with self.assertRaisesRegex(ValueError, 'a1 -> a1'):
            validate_three_phase_trigger_training_config(ThreePhaseTriggerTrainingConfig(**raw), '<x>')

    def test_validation_steps_duplicates_are_rejected_before_canonicalization(self):
        raw = self._raw()
        raw['validation'] = {'enabled': False, 'steps': [5, 5]}
        config = ThreePhaseTriggerTrainingConfig(**raw)
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_three_phase_trigger_training_config(config, '<x>')


if __name__ == '__main__':
    unittest.main()
