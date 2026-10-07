# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
import hashlib
import random
import numpy as np
import torch
SAMPLER_VERSION = 'uniform-nonzero-gt-pixel-keyed-sha256-v1'

def random_foreground_points(masks: torch.Tensor, *, dataset: str, image_id: int, annotation_ids: list[int], seed: int) -> torch.Tensor:
    """Uniformly sample one (x,y) pixel per resized GT mask, returning CPU [N,1,2].

    The candidate set is mask > 0, matching EdgeSAM's mask-rand evaluation.
    Each annotation uses its own RNG, so chunk size, resume and model choice do
    not change the point. The global Python/NumPy/Torch RNGs are not consumed.
    """
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(masks) != len(annotation_ids):
        raise ValueError('Expected one [H,W] GT mask per annotation ID')
    if seed < 0:
        raise ValueError('point-seed must be nonnegative')
    arrays = masks.detach().cpu().numpy()
    width = arrays.shape[-1]
    points = []
    for (mask, annotation_id) in zip(arrays, annotation_ids):
        candidates = np.flatnonzero(mask > 0)
        if not len(candidates):
            raise ValueError(f'Empty GT mask: {dataset}/{image_id}/{annotation_id}')
        key = f'{SAMPLER_VERSION}:{seed}:{dataset}:{image_id}:{annotation_id}'.encode('ascii')
        rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big'))
        pixel = int(candidates[rng.randrange(len(candidates))])
        points.append((pixel % width, pixel // width))
    return torch.tensor(points, dtype=torch.int64).reshape(-1, 1, 2)
