import unittest

import torch

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds


class AdvancedPromptEmbedsMetadataTest(unittest.TestCase):
    def test_tensor_transforms_preserve_nested_runtime_metadata(self):
        embeds = AdvancedPromptEmbeds(
            text_embeds=[torch.ones(2, requires_grad=True)],
            trigger_runtime_metadata={
                'virtual_tokens': 4,
                'indices': [torch.tensor([1, 2]), {'phase': 'a1'}],
            },
        )

        detached = embeds.detach()
        moved = embeds.to(dtype=torch.float64)
        cloned = embeds.clone()

        self.assertFalse(detached.text_embeds[0].requires_grad)
        self.assertEqual(detached.trigger_runtime_metadata[0]['virtual_tokens'], 4)
        self.assertEqual(moved.text_embeds[0].dtype, torch.float64)
        self.assertEqual(moved.trigger_runtime_metadata[0]['indices'][1]['phase'], 'a1')
        self.assertIsNot(cloned.trigger_runtime_metadata[0], embeds.trigger_runtime_metadata[0])
        torch.testing.assert_close(cloned.text_embeds[0], embeds.text_embeds[0])


if __name__ == '__main__':
    unittest.main()
