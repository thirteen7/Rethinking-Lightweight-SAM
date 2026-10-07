"""Fast CPU tests for prompt encoding, sampling and illegal checkpoint gates."""
import tempfile
import unittest
from pathlib import Path
import torch
from prompt_adaptive_sam.model import load_payload,FORMAT
from prompt_adaptive_sam.datasets import selected_indices
from scripts.side_coco_eval.prompt_contract import prompt_contract

class Contracts(unittest.TestCase):
    def test_cap_is_without_replacement_and_keeps_original_index_seed(self):
        a=selected_indices(100,7)
        expected=torch.randperm(100,generator=torch.Generator().manual_seed(7))[:64].tolist()
        self.assertEqual(a,expected)
        self.assertEqual(len(set(a)),64)
        self.assertNotEqual(a,selected_indices(100,8))
        self.assertEqual(selected_indices(4,7),[0,1,2,3])

    def test_ignored_token_zero_and_padding_token_distinct(self):
        from tinysam import sam_model_registry
        sam=sam_model_registry['vit_t']()
        points=torch.tensor([[[20.,30.],[200.,100.],[50.,60.]]])
        labels=torch.tensor([[1.,-2.,-1.]])
        sparse=sam.prompt_encoder._embed_points(points,labels,pad=False)
        self.assertEqual(tuple(sparse.shape),(1,3,256))
        self.assertTrue(torch.equal(sparse[:,1],torch.zeros_like(sparse[:,1])))
        self.assertTrue(torch.equal(sparse[:,2],sam.prompt_encoder.not_a_point_embed.weight))

    def test_incomplete_and_rescoring_checkpoints_are_rejected(self):
        valid=dict(format=FORMAT,model='tinysam',added_quality_head=False,rescoring=False,
            prompt_contract=prompt_contract(),complete_training=True,smoke=False)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pth'
            for override in (dict(complete_training=False),dict(smoke=True),dict(rescoring=True),dict(added_quality_head=True),dict(prompt_contract={})):
                torch.save(dict(valid,**override),path)
                with self.assertRaises(ValueError): load_payload(path)

if __name__=='__main__': unittest.main()
