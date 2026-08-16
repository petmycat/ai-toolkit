import os
import tempfile
import unittest
from types import SimpleNamespace

from PIL import Image

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class SemanticScaffoldProbeSourceTest(unittest.TestCase):
    def test_manifest_heldout_items_are_loaded_outside_filtered_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            for name, color in [('train.png', 'red'), ('heldout.png', 'blue')]:
                Image.new('RGB', (32, 32), color).save(os.path.join(root, name))
            trainer = object.__new__(SDTrainer)
            trainer.three_phase_trigger_training = SimpleNamespace(
                data_split=SimpleNamespace(manifest_path=os.path.join(root, 'split.json'))
            )
            with open(trainer.three_phase_trigger_training.data_split.manifest_path, 'w', encoding='utf-8') as handle:
                handle.write('{"train_item_ids": ["train.png"], "heldout_item_ids": ["heldout.png"]}')
            trainer.data_loader = SimpleNamespace(dataset=SimpleNamespace(
                datasets=[SimpleNamespace(
                    dataset_path=root,
                    file_list=[SimpleNamespace(
                        dataset_relative_item_id='train.png',
                        path=os.path.join(root, 'train.png'),
                        caption_template='a train',
                        raw_caption='',
                    )]
                )]
            ))
            _, items = trainer._semantic_scaffold_items()
            self.assertEqual({item['dataset_relative_item_id'] for item in items}, {'train.png', 'heldout.png'})
            self.assertEqual(next(item for item in items if item['dataset_relative_item_id'] == 'heldout.png')['image_path'], os.path.join(root, 'heldout.png'))


if __name__ == '__main__':
    unittest.main()
