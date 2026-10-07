"""The public pixel-coordinate API must supply float32 decoder prompts."""
import unittest
import numpy as np
import torch
from prompt_adaptive_sam import Predictor
from qa_sam.utils.transforms import ResizeLongestSide


class RecordingModel:
    def predict(self, embedding, image, coords, labels, hw, nh, prompt, boxes, previous):
        self.coords, self.labels, self.boxes, self.previous = coords, labels, boxes, previous
        return torch.zeros((1, *nh), dtype=torch.bool), torch.zeros(1, dtype=torch.long), torch.zeros((1, 1, 256, 256))


class InteractiveAPIContracts(unittest.TestCase):
    def setUp(self):
        self.predictor = Predictor.__new__(Predictor)
        self.predictor.model = RecordingModel()
        self.predictor.device = torch.device('cpu')
        self.predictor.transform = ResizeLongestSide(1024)
        self.predictor.native_hw = (100, 200)
        self.predictor.input_hw = (512, 1024)
        self.predictor.embedding = torch.zeros(1)
        self.predictor.image = torch.zeros(1)

    def test_native_point_coordinates_are_float32_and_scaled(self):
        mask, logits, choice = self.predictor.predict([[25., 30.]], [1])
        model = self.predictor.model
        self.assertEqual(model.coords.dtype, torch.float32)
        self.assertEqual(model.labels.dtype, torch.float32)
        torch.testing.assert_close(model.coords, torch.tensor([[[128., 153.6]]]))
        self.assertEqual(mask.shape, (100, 200))
        self.assertEqual(logits.shape, (256, 256))
        self.assertEqual(choice, 0)

    def test_box_coordinates_and_previous_logits_are_float32(self):
        self.predictor.predict(box=[5., 10., 110., 70.], previous=np.zeros((256, 256), dtype=np.float64))
        model = self.predictor.model
        self.assertEqual(model.coords.dtype, torch.float32)
        self.assertEqual(model.boxes.dtype, torch.float32)
        self.assertEqual(model.previous.dtype, torch.float32)
        torch.testing.assert_close(model.boxes, torch.tensor([[25.6, 51.2, 563.2, 358.4]]))


if __name__ == '__main__':
    unittest.main()
