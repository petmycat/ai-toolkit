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

    def test_tap_gate_and_ramp(self):
        curriculum = TapCurriculumState()
        self.assertEqual(curriculum.observe_gate(
            step=25, progress=0.5, semantic_cosine=0.8, relative_rms=0.5,
            min_step=25, min_progress=0.2, min_semantic_cosine=0.5,
            min_relative_rms=0.2, patience=2, max_wait_step=50,
        ), 'waiting')
        self.assertEqual(curriculum.observe_gate(
            step=26, progress=0.5, semantic_cosine=0.8, relative_rms=0.5,
            min_step=25, min_progress=0.2, min_semantic_cosine=0.5,
            min_relative_rms=0.2, patience=2, max_wait_step=50,
        ), 'unlocked')
        self.assertEqual(curriculum.advance_ramp(step=31, ramp_steps=10), 0.5)
        self.assertEqual(curriculum.advance_ramp(step=36, ramp_steps=10), 1.0)

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
