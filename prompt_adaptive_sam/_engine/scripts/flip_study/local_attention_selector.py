# Extracted from the verified research implementation; see LICENSE and NOTICE.
from scripts.flip_study.candidate_selector import GainSelector
import torch
from torch import nn
KINDS = ['global_risk', 'local_pool', 'local_attention']

def local_grid(coords, hw):
    """75 locations: 5x5 windows of 32/96/256 input pixels around first prompt."""
    offsets = torch.linspace(-0.5, 0.5, 5, device=coords.device, dtype=coords.dtype)
    (yy, xx) = torch.meshgrid(offsets, offsets, indexing='ij')
    xy = torch.stack([xx, yy], -1).reshape(1, 1, 25, 2)
    scales = coords.new_tensor([32.0, 96.0, 256.0]).reshape(1, 3, 1, 1)
    locations = coords[:, :1, None, :] + xy * scales
    valid = ((locations[..., 0] >= 0) & (locations[..., 0] < hw[1]) & (locations[..., 1] >= 0) & (locations[..., 1] < hw[0])).reshape(len(coords), 75)
    relative = torch.cat([(xy * scales / 256).expand(len(coords), -1, -1, -1), (scales / 256).expand(len(coords), -1, 25, -1)], -1).reshape(len(coords), 75, 3)
    assert valid.any(-1).all(), 'Prompts must intersect the valid image'
    return (2 * (locations.reshape(len(coords), 75, 1, 2) + 0.5) / 1024 - 1, valid, relative)

class LocalSelector(GainSelector):

    def __init__(self, kind):
        super().__init__()
        assert kind in KINDS
        self.kind = kind
        self.register_buffer('context_mean', torch.zeros(42))
        self.register_buffer('context_std', torch.ones(42))
        if kind != 'global_risk':
            self.context_projection = nn.Sequential(nn.Linear(42, 32), nn.GELU(), nn.LayerNorm(32))
            self.query_norm = nn.LayerNorm(32)
            self.local_mix = nn.MultiheadAttention(32, 4, dropout=0, batch_first=True) if kind == 'local_attention' else nn.Linear(32, 32)
            self.ffn = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))

    def forward(self, tokens, stats, context=None, valid=None):
        z = self.projection(tokens)
        if self.kind != 'global_risk':
            memory = self.context_projection((context - self.context_mean) / self.context_std)
            if self.kind == 'local_attention':
                (update, _) = self.local_mix(self.query_norm(z), memory, memory, key_padding_mask=~valid, need_weights=False)
            else:
                weights = valid.to(memory.dtype)[..., None]
                update = self.local_mix((memory * weights).sum(1) / weights.sum(1).clamp_min(1))[:, None]
            z = z + update
            z = z + self.ffn(z)
        base = z[:, :1].expand(-1, 3, -1)
        return self.mlp(torch.cat([z[:, 1:], base, z[:, 1:] - base, (stats - self.stat_mean) / self.stat_std], -1))
