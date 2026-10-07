"""Small interactive API with original-image pixel coordinates."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from .model import load_model
from qa_sam.utils.transforms import ResizeLongestSide


class Predictor:
    def __init__(self,checkpoint,device='cpu'):
        self.model = load_model(checkpoint,device)
        self.device = torch.device(device)
        self.transform = ResizeLongestSide(1024)
        self.embedding = None

    @torch.inference_mode()
    def set_image(self,image_rgb):
        image_rgb = np.asarray(image_rgb,dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3: raise ValueError('Expected RGB HxWx3')
        self.native_hw = tuple(image_rgb.shape[:2])
        resized = self.transform.apply_image(image_rgb)
        self.input_hw = tuple(resized.shape[:2])
        image = torch.as_tensor(resized,device=self.device).permute(2,0,1).float()[None]
        image = (image-self.model.sam.pixel_mean)/self.model.sam.pixel_std
        self.image = F.pad(image,(0,1024-self.input_hw[1],0,1024-self.input_hw[0]))
        self.embedding = self.model.sam.image_encoder(self.image)

    @torch.inference_mode()
    def predict(self,point_coords=None,point_labels=None,box=None,previous=None):
        """One object; return boolean mask and logits for the next point round.

        Include the full point history on subsequent rounds. Box prompts have
        at most two corrective clicks. Coordinates and boxes use native pixels.
        """
        if self.embedding is None: raise RuntimeError('Call set_image first')
        coords = np.asarray(point_coords if point_coords is not None else [],dtype=np.float32).reshape(-1,2)
        labels = np.asarray(point_labels if point_labels is not None else [],dtype=np.float32).reshape(-1)
        if len(coords) != len(labels): raise ValueError('Point/label lengths differ')
        if box is None and not len(coords): raise ValueError('Supply a point or a box')
        if len(coords) and not np.isin(labels,[-2,-1,0,1]).all(): raise ValueError('Invalid point label')
        coords = self.transform.apply_coords(coords,self.native_hw)
        coords = torch.as_tensor(coords,device=self.device)[None]
        labels = torch.as_tensor(labels,device=self.device)[None]
        boxes = None
        if box is not None:
            boxes = self.transform.apply_boxes(np.asarray(box,dtype=np.float32).reshape(1,4),self.native_hw)
            boxes = torch.as_tensor(boxes,device=self.device)
        if previous is not None:
            previous = torch.as_tensor(previous,device=self.device,dtype=torch.float32).reshape(1,1,256,256)
        mask,choice,low = self.model.predict(self.embedding,self.image,coords,labels,
            self.input_hw,self.native_hw,'box' if boxes is not None else 'point',boxes,previous)
        return mask[0].cpu().numpy(),low[0,0].cpu().numpy(),int(choice[0])
