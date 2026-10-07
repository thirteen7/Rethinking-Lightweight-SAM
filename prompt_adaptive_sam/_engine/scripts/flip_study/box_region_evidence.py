# Extracted from the verified research implementation; see LICENSE and NOTICE.
import torch
from torch import nn
from torch.nn import functional as F
VARIANTS = ('whole', 'edit', 'zero')
REGION_DIM = 32

@torch.no_grad()
def region_features(embedding, lows, hw, variant, canvas_size=1024):
    """Return [N,4,32]; lows are [actual-current,new0..3] at fixed prompts.

    One embedding [1,C,H,W] is shared by all objects of an image; [N,C,H,W]
    also works for prepacked caches. Binary masks use >0; padding is excluded
    using inference-known hw and low-grid pixel centers. Empty pooling is zero.
    """
    assert variant in VARIANTS and lows.ndim == 4 and (lows.shape[1] == 5)
    assert embedding.ndim == 4 and embedding.shape[0] in (1, len(lows))
    assert embedding.shape[1] >= 14 and embedding.device == lows.device
    assert torch.isfinite(embedding).all() and torch.isfinite(lows).all()
    (n, _, h, w) = lows.shape
    assert 0 < hw[0] <= canvas_size and 0 < hw[1] <= canvas_size
    if variant == 'zero':
        return lows.new_zeros(n, 4, REGION_DIM)
    yy = (torch.arange(h, device=lows.device) + 0.5) * canvas_size / h
    xx = (torch.arange(w, device=lows.device) + 0.5) * canvas_size / w
    valid = (yy[:, None] < hw[0]) & (xx[None, :] < hw[1])
    foreground = (lows.detach() > 0) & valid
    (current, new) = (foreground[:, :1], foreground[:, 1:])
    whole = torch.stack([new, current.expand_as(new)], 2)
    edit = torch.stack([new & ~current, current & ~new], 2)
    regions = whole if variant == 'whole' else edit
    mass = regions.sum((-1, -2)).float()
    fraction = mass / valid.sum().clamp_min(1)
    empty = mass == 0
    c = embedding.shape[1]
    spatial = embedding.detach().float()
    mapped = torch.stack([spatial[:, i * c // 14:(i + 1) * c // 14].mean(1) for i in range(14)], 1)
    mapped = mapped.expand(n, -1, -1, -1)
    weights = F.interpolate(regions.reshape(n, 8, h, w).float(), mapped.shape[-2:], mode='area')
    denominator = weights.sum((-1, -2))
    pooled = torch.einsum('nrhw,nchw->nrc', weights, mapped)
    pooled = pooled / denominator.clamp_min(1e-12)[..., None]
    pooled = pooled.masked_fill((denominator == 0)[..., None], 0).reshape(n, 4, 2, 14)
    result = torch.cat([pooled, fraction[..., None], empty.float()[..., None]], -1)
    result = result.reshape(n, 4, REGION_DIM).to(lows.dtype)
    assert torch.isfinite(result).all()
    return result

class RegionSelector(nn.Module):
    """52,471 parameters for all variants, including nominal zero32 capacity.

    Preserve LayerNorm(210) for the old core; separate LayerNorm(32) adds the
    same 64 parameters as expanding its size. Zero extra Linear columns make
    all variants initially ignore spatial evidence with identical predictions.
    """

    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(256, 48), nn.GELU(), nn.LayerNorm(48))
        self.mlp = nn.Sequential(nn.LayerNorm(210), nn.Linear(242, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        self.region_norm = nn.LayerNorm(REGION_DIM)
        self.register_buffer('stat_mean', torch.zeros(64))
        self.register_buffer('stat_std', torch.ones(64))
        self.register_buffer('box_mean', torch.zeros(17))
        self.register_buffer('box_std', torch.ones(17))

    @classmethod
    def from_base_state(cls, state):
        head = cls()
        expected = head.state_dict()
        for (key, value) in state.items():
            assert key in expected, key
            if key in ('stat_mean', 'stat_std'):
                assert value.shape == (32,)
                expected[key][:32].copy_(value)
            elif key == 'mlp.1.weight':
                assert value.shape == (128, 210)
                expected[key][:, :210].copy_(value)
                expected[key][:, 210:].zero_()
            else:
                assert expected[key].shape == value.shape, key
                expected[key].copy_(value)
        assert set(state) == set(expected) - {'region_norm.weight', 'region_norm.bias'}
        head.load_state_dict(expected, strict=True)
        assert sum((p.numel() for p in head.parameters())) == 52471
        return head

    def forward(self, tokens, stats, box):
        assert tokens.shape[1:] == (4, 256) and stats.shape[1:] == (3, 64)
        assert box.shape[1:] == (4, 17)
        z = self.project(tokens)
        (a, base) = (z[:, 1:], z[:, :1].expand(-1, 3, -1))
        s = ((stats - self.stat_mean) / self.stat_std).clamp(-10, 10)
        b = ((box - self.box_mean) / self.box_std).clamp(-10, 10)
        core = torch.cat([a, base, a - base, s[..., :32], b[:, 1:], b[:, 1:] - b[:, :1]], -1)
        combined = torch.cat([self.mlp[0](core), self.region_norm(s[..., 32:])], -1)
        return self.mlp[1:](combined)

def pair_prediction(head, tokens, stats, geometry):
    """Exact old 3+1 packing; padding new3 does not introduce extra candidates."""
    assert tokens.shape[1:] == (2, 4, 256)
    assert stats.shape[1:] == (2, 3, 64) and geometry.shape[1:] == (2, 4, 17)
    first = head(tokens[:, 0], stats[:, 0], geometry[:, 0])
    last = head(tokens[:, 1], stats[:, 1], geometry[:, 1])[:, :1]
    return torch.cat([first, last], 1)

def training_loss(prediction, iou, native_iou, compatible):
    """Original gain/harm/regret objective; GT is a detached label only.

    This optional cache-training interface mirrors adapt_box_current_anchor.
    Production wrappers may reuse that immutable trainer with pair_prediction.
    """
    assert prediction.shape[1:] == (4, 3) and compatible.shape == prediction.shape[:2]
    assert iou.shape == native_iou.shape == (len(prediction), 5)
    legacy = (100 * (iou[:, 1:] - iou[:, :1])).detach().float()
    native = (100 * (native_iou[:, 1:] - native_iou[:, :1])).detach().float()
    utility = torch.minimum(legacy, native)
    assert torch.isfinite(utility).all()
    loss = 0.5 * F.mse_loss(prediction[..., 0] / 10, legacy / 10)
    loss += 0.5 * F.mse_loss(prediction[..., 1] / 10, native / 10)
    loss += 0.1 * F.binary_cross_entropy_with_logits(prediction[..., 2], (utility < -0.5).float())
    logits = torch.cat([prediction.new_zeros(len(prediction), 1), prediction[..., :2].min(-1).values], 1)
    eligible = torch.cat([torch.ones(len(prediction), 1, dtype=torch.bool, device=prediction.device), compatible], 1)
    truth = torch.cat([prediction.new_zeros(len(prediction), 1), utility], 1)
    regret = truth.masked_fill(~eligible, -torch.inf).max(1).values
    regret -= (logits.masked_fill(~eligible, -torch.inf).softmax(1) * truth).sum(1)
    return loss + 0.02 * regret.mean()
