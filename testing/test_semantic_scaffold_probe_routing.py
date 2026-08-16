import unittest

from types import SimpleNamespace


class SemanticScaffoldProbeRoutingTest(unittest.TestCase):
    def test_all_scope_case_can_be_routed_by_manifest_membership(self):
        manifest_ids = {
            'train': {'train/a.png'},
            'heldout': {'heldout/b.png'},
        }
        cases = [
            SimpleNamespace(split='all', item_id='train/a.png'),
            SimpleNamespace(split='all', item_id='heldout/b.png'),
        ]
        routed = {'train': [], 'heldout': []}
        for case in cases:
            split = next(name for name, ids in manifest_ids.items() if case.item_id in ids)
            routed[split].append(case.item_id)
        self.assertEqual(routed, {
            'train': ['train/a.png'],
            'heldout': ['heldout/b.png'],
        })


if __name__ == '__main__':
    unittest.main()
