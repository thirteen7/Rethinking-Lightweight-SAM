# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
import torch
from torch.nn import functional as F
from qa_sam.utils.common import distance_transform

@torch.inference_mode()
def legacy_centroid_batched(masks: torch.Tensor) -> torch.Tensor:
    """The legacy distance-transform maximum, computed for a mask batch."""
    if masks.ndim != 4 or masks.shape[1] != 1 or masks.dtype != torch.bool:
        raise ValueError('Expected boolean masks [N,1,H,W]')
    width = masks.shape[-1]
    transformed = distance_transform((~F.pad(masks, (1, 1, 1, 1), value=0)).float())
    indices = transformed[:, :, 1:-1, 1:-1].flatten(1).argmax(1)
    return torch.stack((indices % width, torch.div(indices, width, rounding_mode='trunc')), dim=-1)[:, None]
