# Extracted from the verified research implementation; see LICENSE and NOTICE.
import torch
from torch.nn import functional as F

def box_features(low, boxes, hw):
    """Prompt geometry and mask/box agreement, [N,4,17], without GT."""
    (n, k, h, w) = low.shape
    assert (h, w) == (256, 256) and boxes.shape == (n, 4)
    xx = (torch.arange(w, device=low.device) + 0.5).view(1, 1, 1, w) * 4
    yy = (torch.arange(h, device=low.device) + 0.5).view(1, 1, h, 1) * 4
    (x0, y0, x1, y1) = [boxes[:, j].view(n, 1, 1, 1) for j in range(4)]
    (bw, bh) = ((x1 - x0).clamp_min(1), (y1 - y0).clamp_min(1))
    valid = (xx < hw[1]) & (yy < hw[0])
    inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1) & valid
    fg = (low > 0) & valid
    count = lambda z: z.flatten(2).sum(-1).float()
    box_area = count(inside).clamp_min(1)
    fg_area = count(fg).clamp_min(1)
    in_area = count(fg & inside)
    margin_x = (bw * 0.07).clamp_min(4)
    margin_y = (bh * 0.07).clamp_min(4)
    edges = [inside & (xx <= x0 + margin_x), inside & (xx >= x1 - margin_x), inside & (yy <= y0 + margin_y), inside & (yy >= y1 - margin_y)]
    edge_fill = [count(fg & edge) / count(edge).clamp_min(1) for edge in edges]
    center = (2 * (boxes[:, :2] + boxes[:, 2:]) / 4 + 1) / 1024 - 1
    center_logits = F.grid_sample(low, center[:, None, None], align_corners=False, padding_mode='border')[:, :, 0, 0]
    centroid_x = (fg.float() * xx).flatten(2).sum(-1) / fg_area
    centroid_y = (fg.float() * yy).flatten(2).sum(-1) / fg_area
    extend = lambda x: x.expand(n, k)
    base = [extend((bw / 1024).view(n, 1)), extend((bh / 1024).view(n, 1)), extend((bw / bh).log().clamp(-5, 5).view(n, 1) / 5), extend((bw * bh / (hw[0] * hw[1])).log().clamp(-10, 1).view(n, 1) / 10)]
    per = [fg_area / box_area, in_area / box_area, in_area / fg_area, (fg_area - in_area) / fg_area, center_logits.clamp(-10, 10) / 10, ((centroid_x - (x0[:, 0, 0, 0] + x1[:, 0, 0, 0])[:, None] / 2) / bw[:, 0, 0, 0, None]).clamp(-2, 2), ((centroid_y - (y0[:, 0, 0, 0] + y1[:, 0, 0, 0])[:, None] / 2) / bh[:, 0, 0, 0, None]).clamp(-2, 2), *edge_fill, (fg_area / box_area).log().clamp(-5, 5) / 5, (in_area / box_area).clamp(0, 1)]
    result = torch.stack(base + per, -1)
    assert result.shape == (n, 4, 17) and torch.isfinite(result).all()
    return result
