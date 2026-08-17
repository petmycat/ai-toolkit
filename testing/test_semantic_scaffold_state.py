import os
import tempfile
import unittest

import torch

from toolkit.semantic_scaffold_state import (
    HelperLatchState,
    SemanticScaffoldState,
    TapCurriculumState,
)


class SemanticScaffoldStateTest(unittest.TestCase):
    def test_ema_observations_are_detached_and_independent(self):
        state = SemanticScaffoldState((0, 3))
        observation = torch.tensor([1.0, 2.0], requires_grad=True)
        stored = state.update_ema('0', observation, decay=0.5, kind='semantic')
        with torch.no_grad():
            observation.add_(1.0)
        self.assertFalse(stored.requires_grad)
        torch.testing.assert_close(state.semantic_prototypes['0'], torch.tensor([1.0, 2.0]))
        self.assertEqual(state.semantic_observation_counts['0'], 1)

    def test_helper_latch_is_permanent(self):
        latch = HelperLatchState()
        self.assertFalse(latch.advance(0.1, margin=0.2, patience=2, step=1))
        self.assertFalse(latch.advance(0.3, margin=0.2, patience=2, step=2))
        self.assertTrue(latch.advance(0.3, margin=0.2, patience=2, step=3))
        self.assertTrue(latch.latched)
        self.assertFalse(latch.advance(-1.0, margin=0.2, patience=2, step=4))

    def test_tap_maturity_gate_handoffs_on_next_step(self):
        curriculum = TapCurriculumState()
        inputs = dict(
            semantic_gain=0.2, helper_cosine=0.6, prototype_loss=0.05,
            gain_drift=0.01, minimum_observations=10, min_step=25,
            min_semantic_gain=0.1, min_helper_cosine=0.2, max_helper_cosine=0.9,
            max_prototype_loss=0.1, max_gain_drift=0.02,
            required_observations=5, patience=2, max_wait_step=50,
        )
        self.assertEqual(curriculum.observe_maturity(step=25, **inputs), 'waiting')
        self.assertEqual(curriculum.observe_maturity(step=26, **inputs), 'handoff_pending')
        self.assertEqual(curriculum.unlock_step, 27)
        self.assertFalse(curriculum.forced_handoff)
        self.assertEqual(curriculum.advance_ramp(step=32, ramp_steps=10), 0.5)
        self.assertEqual(curriculum.advance_ramp(step=37, ramp_steps=10), 1.0)

    def test_max_wait_forces_handoff_without_failure(self):
        curriculum = TapCurriculumState()
        result = curriculum.observe_maturity(
            step=10, semantic_gain=-1.0, helper_cosine=1.0,
            prototype_loss=2.0, gain_drift=1.0, minimum_observations=1,
            min_step=5, min_semantic_gain=0.1, min_helper_cosine=0.2,
            max_helper_cosine=0.9, max_prototype_loss=0.1,
            max_gain_drift=0.02, required_observations=5,
            patience=2, max_wait_step=10,
        )
        self.assertEqual(result, 'forced_handoff_pending')
        self.assertEqual(curriculum.unlock_step, 11)
        self.assertTrue(curriculum.forced_handoff)
        self.assertFalse(curriculum.failure)
        self.assertIn('semantic_gain', curriculum.unmet_conditions)

    def test_empty_state_round_trip(self):
        state = SemanticScaffoldState((0, 3))
        with tempfile.TemporaryDirectory() as directory:
            json_path = os.path.join(directory, 'state.json')
            tensor_path = os.path.join(directory, 'state.safetensors')
            state.save(json_path, tensor_path)
            loaded = SemanticScaffoldState.load(json_path, tensor_path)
        self.assertEqual(loaded.tap_layers, (0, 3))
        self.assertEqual(loaded.semantic_prototypes, {})
        self.assertEqual(loaded.tap_prototypes, {})

    def test_state_round_trip(self):
        state = SemanticScaffoldState((0, 3))
        state.update_ema('0', torch.tensor([1.0, 2.0]), decay=0.5, kind='semantic')
        state.gain_ema = 0.25
        state.helper_gain_ema['h'] = 0.3
        with tempfile.TemporaryDirectory() as directory:
            json_path = f'{directory}/state.json'
            tensor_path = f'{directory}/state.pt'
            state.save(json_path, tensor_path)
            restored = SemanticScaffoldState.load(json_path, tensor_path)
        torch.testing.assert_close(restored.semantic_prototypes['0'], state.semantic_prototypes['0'])
        self.assertEqual(restored.gain_ema, state.gain_ema)
        self.assertEqual(restored.tap_layers, state.tap_layers)


if __name__ == '__main__':
    unittest.main()
