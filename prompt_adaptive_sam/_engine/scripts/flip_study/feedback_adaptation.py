# Extracted from the verified research implementation; see LICENSE and NOTICE.
from scripts.flip_study.local_attention_selector import LocalSelector
import torch
from torch import nn
from torch.nn import functional as F
TEMPORAL_DIM = 24

def latest_click(coords, labels):
    valid = (labels == 0) | (labels == 1)
    indices = torch.arange(labels.shape[1], device=coords.device)[None]
    last = torch.where(valid, indices, -1).amax(1).clamp_min(0)
    return (coords[torch.arange(len(coords), device=coords.device), last][:, None], last)

def point_violations(low, coords, labels):
    grid = (2 * (coords + 0.5) / 1024 - 1)[:, :, None]
    values = F.grid_sample(low, grid, align_corners=False, padding_mode='border')[:, :, :, 0]
    valid = (labels == 0) | (labels == 1)
    return ((values > 0) != (labels == 1)[:, None]) & valid[:, None]

def temporal_features(low, previous, coords, labels, hw):
    """Twelve history descriptors per candidate, and the same for candidate 0."""
    (latest, last) = latest_click(coords, labels)
    (yy, xx) = torch.meshgrid(torch.arange(256, device=low.device), torch.arange(256, device=low.device), indexing='ij')
    valid = ((yy + 0.5) * 4 < hw[0]) & ((xx + 0.5) * 4 < hw[1])
    count = valid.sum().clamp_min(1)
    (fg, old) = ((low > 0) & valid, (previous > 0) & valid)
    total = lambda x: x.flatten(2).sum(-1).float()
    (area, old_area) = (total(fg), total(old))
    (inter, union) = (total(fg & old), total(fg | old))
    changed = (fg != old) & valid
    d2 = (xx[None] * 4 - latest[:, 0, 0, None, None]) ** 2 + (yy[None] * 4 - latest[:, 0, 1, None, None]) ** 2
    remote = (d2 > 128 ** 2)[:, None] & valid
    grid = (2 * (latest + 0.5) / 1024 - 1)[:, :, None]
    now_point = F.grid_sample(low, grid, align_corners=False, padding_mode='border')[:, :, 0, 0]
    old_point = F.grid_sample(previous, grid, align_corners=False, padding_mode='border')[:, :, 0, 0]
    sign = (2 * labels[torch.arange(len(labels), device=labels.device), last] - 1)[:, None]
    changed_count = total(changed).clamp_min(1)
    uncertainty = total(changed * (previous.abs() <= 1)) / changed_count
    margin = total(changed * low.abs().clamp_max(10)) / (10 * changed_count)
    remote_confident = total(changed & remote & (previous.abs() > 3)) / count
    distance = ((latest[:, 0] - coords[:, 0]) ** 2).sum(-1).sqrt()[:, None] / 1024
    f = torch.stack([inter / union.clamp_min(1), ((area + 1) / (old_area + 1)).log().clamp(-5, 5) / 5, inter / old_area.clamp_min(1), total(fg & ~old) / area.clamp_min(1), total(changed) / count, total(changed & remote) / count, (now_point * sign).clamp(-10, 10) / 10, (old_point * sign).clamp(-10, 10).expand_as(area) / 10, uncertainty, margin, remote_confident, distance.expand_as(area)], -1)
    return torch.cat([f[:, 1:], f[:, :1].expand(-1, 3, -1)], -1)

class TemporalSelector(LocalSelector):
    """A zero-initialized correction to the source head's gain/risk estimates.

    Existing projections attend to a second visual neighborhood at the latest
    valid click. Candidate differences and previous-mask descriptors predict
    corrections to the original scores; no extra SAM decoding is required.
    """

    def __init__(self):
        super().__init__('local_attention')
        self.register_buffer('temporal_mean', torch.zeros(TEMPORAL_DIM))
        self.register_buffer('temporal_std', torch.ones(TEMPORAL_DIM))
        self.temporal_mlp = nn.Sequential(nn.Linear(64 + TEMPORAL_DIM + 2, 48), nn.GELU(), nn.Linear(48, 2))
        nn.init.zeros_(self.temporal_mlp[-1].weight)
        nn.init.zeros_(self.temporal_mlp[-1].bias)

    def forward(self, tokens, stats, context, valid, latest_context, latest_valid, temporal):
        original = super().forward(tokens, stats, context, valid)
        memory = self.context_projection((latest_context - self.context_mean) / self.context_std)
        query = self.query_norm(self.projection(tokens))
        (local, _) = self.local_mix(query, memory, memory, key_padding_mask=~latest_valid, need_weights=False)
        base = local[:, :1].expand(-1, 3, -1)
        history = ((temporal - self.temporal_mean) / self.temporal_std).clamp(-10, 10)
        extra = torch.cat([local[:, 1:] - base, base, history, original], -1)
        return original + self.temporal_mlp(extra)
