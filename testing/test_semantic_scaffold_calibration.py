import os
import tempfile
import unittest

import torch

from toolkit.semantic_scaffold_calibration import (
    SemanticScaffoldCalibrationError,
    build_fixed_probe_cases,
    helper_gain,
    run_calibration,
    select_helpers,
)


class SemanticScaffoldCalibrationTest(unittest.TestCase):
    def _manifest(self):
        return {'train_item_ids': ['train.png'], 'heldout_item_ids': ['heldout.png']}

    def _items(self):
        return [
            {'dataset_relative_item_id': 'train.png', 'caption': 'a train'},
            {'dataset_relative_item_id': 'heldout.png', 'caption': 'a heldout'},
        ]

    def test_all_scope_uses_every_item_without_split_requirements(self):
        cases = build_fixed_probe_cases(
            self._items(), {'train_item_ids': ['train.png'], 'heldout_item_ids': []},
            noise_seeds=[3], fixed_timesteps=[10], probe_scope='all'
        )
        self.assertEqual([case.split for case in cases], ['all', 'all'])

    def test_probe_cases_are_shared_by_phrase_and_deterministic(self):
        first = build_fixed_probe_cases(
            self._items(), self._manifest(), noise_seeds=[3], fixed_timesteps=[10]
        )
        second = build_fixed_probe_cases(
            self._items(), self._manifest(), noise_seeds=[3], fixed_timesteps=[10]
        )
        self.assertEqual([case.as_dict() for case in first], [case.as_dict() for case in second])
        self.assertEqual(len(first), 2)
        self.assertNotEqual(first[0].probe_case_id, first[1].probe_case_id)

    def test_equal_helper_and_neutral_predictions_have_zero_gain(self):
        prediction = torch.tensor([1.0, 2.0])
        target = torch.zeros_like(prediction)
        self.assertEqual(helper_gain(prediction, prediction, target), 0.0)

    def test_conditioning_space_selection_can_keep_visually_relevant_helper(self):
        result = select_helpers(
            {
                'Ghibli anime': [
                    {'split': 'all', 'gain': -0.1, 'conditioning_relative_rms': 0.2, 'conditioning_direction_consistency': 0.7},
                    {'split': 'all', 'gain': -0.05, 'conditioning_relative_rms': 0.3, 'conditioning_direction_consistency': 0.8},
                ]
            },
            min_median_gain=0.0, min_positive_fraction=0.5,
            min_heldout_positive_fraction=0.5, min_p10_gain=-1.0,
            max_train_heldout_gap=1.0, max_helpers=1, sampling_floor=0.05,
            selection_mode='conditioning_space', min_conditioning_relative_rms=0.1,
            min_conditioning_direction_consistency=0.5, min_mean_gain=-0.25,
            max_mean_gain_regression=0.25,
        )
        self.assertEqual(result['selected_helpers'], ['Ghibli anime'])

    def test_helper_selection_requires_both_splits_and_applies_thresholds(self):
        records = {
            'good': [
                {'split': 'train', 'gain': 0.5},
                {'split': 'heldout', 'gain': 0.4},
            ],
            'bad': [
                {'split': 'train', 'gain': -0.5},
                {'split': 'heldout', 'gain': -0.4},
            ],
        }
        result = select_helpers(
            records,
            min_median_gain=0.0,
            min_positive_fraction=0.5,
            min_heldout_positive_fraction=0.5,
            min_p10_gain=-1.0,
            max_train_heldout_gap=0.5,
            max_helpers=1,
            sampling_floor=0.05,
        )
        self.assertEqual(result['selected_helpers'], ['good'])
        self.assertAlmostEqual(sum(result['sampling_weights'].values()), 1.0)

    def test_calibration_writes_manifests_and_rejects_empty_bank(self):
        cases = build_fixed_probe_cases(
            self._items(), self._manifest(), noise_seeds=[3], fixed_timesteps=[10]
        )

        def prepare_case(case):
            return {'target': torch.zeros(2), 'case': case}

        def predict_phrase(phrase, prepared):
            value = 0.5 if phrase == 'good' else 2.0
            return torch.full_like(prepared['target'], value)

        with tempfile.TemporaryDirectory() as directory:
            manifest = run_calibration(
                cases,
                ['good'],
                neutral_phrase='neutral',
                prepare_case=prepare_case,
                predict_phrase=predict_phrase,
                selection_kwargs={
                    'min_median_gain': 0.0,
                    'min_positive_fraction': 0.5,
                    'min_heldout_positive_fraction': 0.5,
                    'min_p10_gain': -1.0,
                    'max_train_heldout_gap': 0.5,
                    'max_helpers': 1,
                    'sampling_floor': 0.05,
                },
                output_dir=directory,
                identity={'model': 'test'},
            )
            self.assertEqual(manifest['selected_helpers'], ['good'])
            self.assertTrue(os.path.isfile(os.path.join(directory, 'semantic_scaffold_manifest.json')))
            self.assertTrue(os.path.isfile(os.path.join(directory, 'semantic_scaffold_probe_manifest.json')))

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SemanticScaffoldCalibrationError):
                run_calibration(
                    cases,
                    ['bad'],
                    neutral_phrase='neutral',
                    prepare_case=prepare_case,
                    predict_phrase=predict_phrase,
                    selection_kwargs={
                        'min_median_gain': 0.9,
                        'min_positive_fraction': 1.0,
                        'min_heldout_positive_fraction': 1.0,
                        'min_p10_gain': 0.9,
                        'max_train_heldout_gap': 0.1,
                        'max_helpers': 1,
                        'sampling_floor': 0.05,
                    },
                    output_dir=directory,
                    identity={'model': 'test'},
                )


if __name__ == '__main__':
    unittest.main()
