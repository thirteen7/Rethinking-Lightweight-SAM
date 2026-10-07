from __future__ import annotations

import torch

from torch import nn

from torch.nn import functional as F

from .exact_decoder import attention_forward, heads, operation

from .dense_decoder import (
    center, factors, native_norm, scale_add_first, scale_add_second,
)

class FactorStore:
    """Real batch64 storage. Common F2/beta2 remain in the per-image engine cache."""
    def __init__(self, device, count=64, image_rows=4096, channels=256, rank=113):
        self.U = torch.empty((count, image_rows, rank), device=device, dtype=torch.float32)
        self.V = torch.empty((count, rank, channels), device=device, dtype=torch.float32)
        self.scale = torch.empty((count, image_rows, 1), device=device, dtype=torch.float32)
        self.hyper = torch.empty((count, 4, 32), device=device, dtype=torch.float32)
        self.head = torch.empty((count, 3), device=device, dtype=torch.float32)
        self.hw = None

    def put(self, start, state):
        n = len(state["U"])
        for name in ("U", "V", "scale", "hyper", "head"):
            getattr(self, name)[start:start+n].copy_(state[name])
        self.hw = state["hw"]

    def take(self, ids):
        return dict(**{name:getattr(self, name)[ids] for name in ("U", "V", "scale", "hyper", "head")}, hw=self.hw)

    def bytes(self):
        return sum(getattr(self, name).numel()*getattr(self, name).element_size() for name in ("U", "V", "scale", "hyper", "head"))

    def layout(self):
        return {name:dict(shape=list(getattr(self, name).shape), stride=list(getattr(self, name).stride()),
                         bytes=getattr(self, name).numel()*4) for name in ("U", "V", "scale", "hyper", "head")}
