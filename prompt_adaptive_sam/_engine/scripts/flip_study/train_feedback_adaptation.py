# Extracted from the verified research implementation; see LICENSE and NOTICE.
from scripts.flip_study.run_candidate_experiment import ious
import torch

@torch.no_grad()
def candidate_scores(sam, low, hw, nh, legacy, native):
    old = torch.empty((len(low), low.shape[1]), dtype=torch.float64, device=low.device)
    new = torch.empty_like(old)
    for k in range(low.shape[1]):
        for start in range(0, len(low), 4):
            end = min(start + 4, len(low))
            mask = sam.postprocess_masks(low[start:end, k:k + 1], hw, nh)[:, 0] > 0
            old[start:end, k] = ious(mask, legacy[start:end])
            new[start:end, k] = ious(mask, native[start:end])
    return (old, new)
