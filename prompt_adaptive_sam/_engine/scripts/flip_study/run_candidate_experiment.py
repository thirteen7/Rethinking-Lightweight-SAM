# Extracted from the verified research implementation; see LICENSE and NOTICE.
import torch

def ious(pred, truth):
    inter = (pred & truth).flatten(1).sum(1)
    union = (pred | truth).flatten(1).sum(1)
    return torch.where(union > 0, inter.double() / union.clamp_min(1), 1.0)
