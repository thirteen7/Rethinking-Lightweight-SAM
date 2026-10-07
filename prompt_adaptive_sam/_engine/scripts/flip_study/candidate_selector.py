# Extracted from the verified research implementation; see LICENSE and NOTICE.
import torch
from torch import nn
from torch.nn import functional as F

def prediction_features(low, quality, coords, labels, input_hw, iteration):
    """Return [objects,3,32] statistics and [objects,3] click compatibility."""
    (n, k, lh, lw) = low.shape
    assert k == 4 and (lh, lw) == (256, 256) and (iteration in (1, 2, 3))
    (h, w) = input_hw
    y = (torch.arange(lh, device=low.device) + 0.5) * 4
    x = (torch.arange(lw, device=low.device) + 0.5) * 4
    valid = (y[:, None] < h) & (x[None, :] < w)
    denom = valid.sum().clamp_min(1)
    foreground = (low > 0) & valid
    flat_sum = lambda value: value.flatten(2).sum(-1)
    area = flat_sum(foreground).float() / denom
    upper = flat_sum((low > 1) & valid)
    lower = flat_sum((low > -1) & valid)
    stability = torch.where(lower > 0, upper.float() / lower.clamp_min(1), 1.0)
    ambiguous = flat_sum((low.abs() <= 1) & valid).float() / denom
    mean = flat_sum(low.clamp(-10, 10) * valid) / (10 * denom)
    original = foreground[:, :1]
    inter = flat_sum(foreground & original)
    union = flat_sum(foreground | original)
    overlap = torch.where(union > 0, inter.float() / union.clamp_min(1), 1.0)
    grid = (2 * (coords + 0.5) / 1024 - 1)[:, :, None, :]
    point_values = F.grid_sample(low, grid, align_corners=False, padding_mode='border')[:, :, :, 0]
    (positive, negative) = (labels == 1, labels == 0)
    violations = ((point_values > 0) != positive[:, None, :]) & (positive | negative)[:, None, :]
    pos_rate = (violations & positive[:, None]).sum(-1).float() / positive.sum(-1)[:, None].clamp_min(1)
    neg_rate = (violations & negative[:, None]).sum(-1).float() / negative.sum(-1)[:, None].clamp_min(1)
    stats = torch.stack([quality, area, stability, ambiguous, mean, pos_rate, neg_rate, overlap], -1)
    compatible = ~(violations[:, 1:] & ~violations[:, :1]).any(-1)
    rounds = low.new_zeros(n, 3, 3)
    rounds[:, :, iteration - 1] = 1
    candidate_ids = torch.eye(4, device=low.device, dtype=low.dtype)[1:][None].expand(n, -1, -1)
    flags = torch.stack([positive.any(-1), negative.any(-1)], -1).to(low.dtype)
    geometry = low.new_tensor([h / 1024, w / 1024])[None].expand(n, -1)
    count = (positive | negative).sum(-1, keepdim=True).to(low.dtype) / 3
    result = torch.cat([stats[:, 1:], stats[:, :1].expand(-1, 3, -1), quality[:, None].expand(-1, 3, -1), rounds, candidate_ids, flags[:, None].expand(-1, 3, -1), geometry[:, None].expand(-1, 3, -1), count[:, None].expand(-1, 3, -1)], -1)
    assert result.shape == (n, 3, 32)
    return (result, compatible)

class GainSelector(nn.Module):

    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(256, 32), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2))
        self.register_buffer('stat_mean', torch.zeros(32))
        self.register_buffer('stat_std', torch.ones(32))

    def forward(self, tokens, stats):
        z = self.projection(tokens)
        base = z[:, :1].expand(-1, 3, -1)
        inputs = torch.cat([z[:, 1:], base, z[:, 1:] - base, (stats - self.stat_mean) / self.stat_std], -1)
        return self.mlp(inputs)

def choose_candidates(prediction, compatible, gate=None):
    """gate=None is the learned ranking ablation; baseline gain is exactly 0."""
    gains = prediction[..., 0]
    if gate is not None:
        if gate.get('reject_all', False):
            return torch.zeros(len(gains), device=gains.device, dtype=torch.long)
        valid = compatible & (gains >= gate['min_gain_pp']) & (prediction[..., 1].sigmoid() <= gate['max_harm_probability'])
    else:
        valid = torch.ones_like(gains, dtype=torch.bool)
    gains = gains.masked_fill(~valid | ~torch.isfinite(gains), -torch.inf)
    return torch.cat([gains.new_zeros(len(gains), 1), gains], -1).argmax(-1)
