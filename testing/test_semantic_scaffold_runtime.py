import unittest

from toolkit.trigger_binding import get_activator_runtime_state


class SemanticScaffoldRuntimeTest(unittest.TestCase):
    def test_semantic_only_enables_embedding_and_internal_only(self):
        state = get_activator_runtime_state('semantic_only')
        self.assertTrue(state.embedding_enabled)
        self.assertTrue(state.internal_enabled)
        self.assertFalse(state.tap_enabled)
        self.assertFalse(state.activator_bypassed)

    def test_existing_modes_keep_their_contract(self):
        bypass = get_activator_runtime_state('activator_bypass')
        self.assertTrue(bypass.activator_bypassed)
        self.assertFalse(bypass.embedding_enabled)
        self.assertFalse(bypass.internal_enabled)
        self.assertFalse(bypass.tap_enabled)
        full = get_activator_runtime_state('full')
        self.assertTrue(full.embedding_enabled)
        self.assertTrue(full.internal_enabled)
        self.assertTrue(full.tap_enabled)


if __name__ == '__main__':
    unittest.main()
