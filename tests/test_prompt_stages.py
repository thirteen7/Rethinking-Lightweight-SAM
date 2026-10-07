"""Incremental comparisons must decode only the requested prompt stage."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from demo.comparison import compare_prompt_stage


class RecordingPredictor:
    def __init__(self):
        self.model = SimpleNamespace(last_trace=None)
        self.calls = []

    def set_image(self, image):
        self.calls.append(('encode', image.shape))

    def predict(self, points, labels, box=None, previous=None):
        self.calls.append(('refined', list(points), previous))
        return np.ones((4, 4), bool), np.full((256, 256), 100 + len(points), np.float32), 1


class PromptStageContracts(unittest.TestCase):
    def setUp(self):
        self.predictor = RecordingPredictor()
        self.image = np.zeros((4, 4, 3), np.uint8)
        self.truth = np.ones((4, 4), bool)
        points = [dict(x=1, y=1, label=1), dict(x=2, y=2, label=1), dict(x=3, y=3, label=1)]
        self.trajectory = [dict(points=points[:size], box=None) for size in (1, 2, 3)]

    def baseline(self, predictor, name, mode, stage, points, labels, box, previous):
        predictor.calls.append(('original', stage, previous))
        return np.eye(4, dtype=bool), np.full((256, 256), stage + 1, np.float32), 0

    def step(self, number, previous=None, name='tinysam'):
        with patch('demo.comparison.original_prediction', side_effect=self.baseline):
            return compare_prompt_stage(self.predictor, name, self.image, self.trajectory[number], self.truth, previous)

    def test_initial_run_never_decodes_future_corrections(self):
        pair = self.step(0)
        self.assertEqual(len(pair['stages']), 1)
        self.assertEqual([call[0] for call in self.predictor.calls], ['encode', 'original', 'refined'])
        self.assertIsNone(self.predictor.calls[1][2])
        self.assertIsNone(self.predictor.calls[2][2])
        self.assertEqual(pair['stages'][0]['original']['iou_percent'], 25)
        self.assertEqual(pair['stages'][0]['refined']['iou_percent'], 100)

    def test_each_added_point_keeps_independent_feedback_and_completed_history(self):
        initial = self.step(0)
        first = self.step(1, initial)
        second = self.step(2, first)
        self.assertEqual([len(pair['stages']) for pair in (initial, first, second)], [1, 2, 3])
        originals = [call for call in self.predictor.calls if call[0] == 'original']
        refined = [call for call in self.predictor.calls if call[0] == 'refined']
        np.testing.assert_array_equal(originals[1][2], np.full((256, 256), 1))
        np.testing.assert_array_equal(refined[1][2], np.full((256, 256), 101))
        np.testing.assert_array_equal(originals[2][2], np.full((256, 256), 2))
        np.testing.assert_array_equal(refined[2][2], np.full((256, 256), 102))
        self.assertEqual(len(initial['stages'][0]['points']), 1)
        self.assertEqual(len(second['stages'][-1]['points']), 3)

    def test_skipping_a_correction_and_changing_model_are_rejected(self):
        initial = self.step(0)
        with self.assertRaisesRegex(ValueError, 'exactly one point'):
            self.step(2, initial)
        with self.assertRaisesRegex(ValueError, 'backbone'):
            self.step(1, initial, 'mobilesam')
        self.assertEqual(len([call for call in self.predictor.calls if call[0] == 'original']), 1)

    def test_completed_target_cannot_add_a_third_correction(self):
        pair = self.step(2, self.step(1, self.step(0)))
        with self.assertRaisesRegex(ValueError, 'two corrections'):
            self.step(2, pair)


if __name__ == '__main__':
    unittest.main()
