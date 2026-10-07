# Extracted from the verified research implementation; see LICENSE and NOTICE.
import torch
from torch import nn
from torch.nn import functional as F
PATCH = 64
CHANNELS = 43

def geometry(coords, labels, hw):
    """Integer-aligned decoder crops; ignored clicks cannot move the crop."""
    valid_click = (labels == 0) | (labels == 1)
    indices = torch.arange(labels.shape[1], device=labels.device)[None]
    last = torch.where(valid_click, indices, -1).amax(1)
    safe = last.clamp_min(0)
    center = coords[torch.arange(len(coords), device=coords.device), safe] / 4
    origin = (center.floor().long() - PATCH // 2).clamp(0, 256 - PATCH)
    (yy, xx) = torch.meshgrid(torch.arange(PATCH, device=coords.device), torch.arange(PATCH, device=coords.device), indexing='ij')
    (x, y) = (origin[:, 0, None, None] + xx, origin[:, 1, None, None] + yy)
    xy = torch.stack([x, y], -1).float()
    distance2 = ((xy - center[:, None, None]) ** 2).sum(-1)
    valid = ((x + 0.5) * 4 < hw[1]) & ((y + 0.5) * 4 < hw[0])
    active = (last > 0) & valid_click[:, -1]
    return dict(x=x, y=y, xy=xy, center=center, last=safe, distance2=distance2, valid=valid[:, None], active=active[:, None, None, None])

def crop(tensor, region):
    tensor = tensor.expand(len(region['x']), -1, -1, -1)
    indices = (region['y'] * 256 + region['x']).flatten(1)
    return tensor.flatten(2).gather(2, indices[:, None].expand(-1, tensor.shape[1], -1)).reshape(len(indices), tensor.shape[1], PATCH, PATCH)

def make_inputs(features, image, base, previous, coords, labels, hw, region):
    rgb = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
    base_patch = crop(base, region)
    previous_patch = crop(previous, region)
    xy = region['xy']
    d2 = ((xy[:, None] - coords[:, :, None, None] / 4) ** 2).sum(-1)
    gaussians = torch.exp(-d2 / (2 * 3.0 ** 2))
    positive = (gaussians * (labels == 1)[:, :, None, None]).amax(1, keepdim=True)
    negative = (gaussians * (labels == 0)[:, :, None, None]).amax(1, keepdim=True)
    last_label = labels[torch.arange(len(labels), device=labels.device), region['last']]
    latest = torch.exp(-region['distance2'] / (2 * 12.0 ** 2))[:, None] * (2 * last_label - 1)[:, None, None, None]
    relative = ((xy - region['center'][:, None, None]) / (PATCH / 2)).permute(0, 3, 1, 2)
    inputs = torch.cat([features, crop(rgb, region), base_patch.clamp(-10, 10) / 10, previous_patch.clamp(-10, 10) / 10, positive, negative, latest, relative, region['valid'].float()], 1)
    assert inputs.shape[1] == CHANNELS
    foreground = (base > 0).float()
    boundary = F.max_pool2d(foreground, 15, 1, 7) - -F.max_pool2d(-foreground, 15, 1, 7) > 0
    support = (region['distance2'][:, None] <= 16 ** 2) | crop(boundary, region)
    support = support & (region['distance2'][:, None] < 32 ** 2) & region['valid'] & region['active']
    taper = ((32 - region['distance2'].sqrt()) / 8).clamp(0, 1)[:, None]
    return (inputs, base_patch, support.float() * taper)

class ClickResidualHead(nn.Module):
    """20k parameters; a zero-initialized residual starts as exact identity."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(CHANNELS, 32, 1), nn.GroupNorm(8, 32), nn.GELU())
        self.body = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(), nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GroupNorm(8, 32), nn.GELU())
        self.output = nn.Conv2d(32, 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs, support):
        z = self.stem(inputs)
        prediction = self.output(z + self.body(z))
        return 8 * prediction[:, :1].tanh() * prediction[:, 1:].sigmoid() * support

def paste(base, delta, region):
    indices = (region['y'] * 256 + region['x']).flatten(1)[:, None]
    return base.flatten(2).scatter_add(2, indices, delta.flatten(2)).reshape_as(base)
