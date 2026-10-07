# Extracted from the verified research implementation; see LICENSE and NOTICE.
import json
import torch
from torch.nn import functional as F

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def loss_fn(prediction, target, area, cfg):
    weight = torch.where(area < 1024, cfg['small_weight'], 1.0).float()[:, None]
    regression = F.smooth_l1_loss(prediction[..., 0], target, reduction='none', beta=1.0)
    harm = F.binary_cross_entropy_with_logits(prediction[..., 1], (target < 0).float(), reduction='none', pos_weight=target.new_tensor(cfg['harm_positive_weight']))
    unsafe = F.smooth_l1_loss(prediction[..., 0].relu(), torch.zeros_like(target), reduction='none', beta=1.0) * (target < -1)
    loss = regression + cfg['harm_weight'] * harm + cfg['unsafe_positive_gain_weight'] * unsafe
    return (loss * weight).sum() / (weight.sum() * 3)
