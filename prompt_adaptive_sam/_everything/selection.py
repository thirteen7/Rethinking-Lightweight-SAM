"""Frozen approximate P2/P4 preview selection; no GT or baseline masks as input."""
from __future__ import annotations
import numpy as np
import torch


def valid_geometry(input_hw, side=64):
    # Native low logit coordinate 4*i has padded input center 4*(4*i+.5)-.5.
    centers = np.arange(side, dtype=np.float32) * 16 + np.float32(1.5)
    return (centers[:, None] < input_hw[0]) & (centers[None, :] < input_hw[1])


def preview_evidence(logits, local_patch, input_hw):
    assert logits.dtype == np.float32 and logits.shape == (1024, 3, 64, 64)
    positive = logits > np.float32(0)
    high, low = logits > np.float32(1), logits > np.float32(-1)
    intersection, union = high.sum((-1, -2)), low.sum((-1, -2))
    with np.errstate(divide="ignore", invalid="ignore"):
        quality = intersection.astype(np.float32) / union.astype(np.float32)
    valid = valid_geometry(input_hw)
    foreground = positive & valid[None, None]
    counts = foreground.sum((-1, -2)).astype(np.int32)
    rows = np.arange(64, dtype=np.float32) * np.float32(4)
    xx, yy = np.meshgrid(rows, rows)
    flat = foreground.reshape(-1, 64, 64)
    boxes = np.stack([np.where(flat, xx, np.inf).min((-1,-2)),
                      np.where(flat, yy, np.inf).min((-1,-2)),
                      np.where(flat, xx, -np.inf).max((-1,-2)),
                      np.where(flat, yy, -np.inf).max((-1,-2))], axis=1).astype(np.float32)
    boxes[counts.reshape(-1) == 0] = 0
    local_max = local_patch.max((-1,-2)).astype(np.float32)
    guard_candidates = (local_max > np.float32(0)) & (counts <= 16)
    pool = (counts.reshape(-1) > 0) & np.isfinite(quality.reshape(-1)) & (quality.reshape(-1) > np.float32(.80))
    return dict(quality=quality, valid_count=counts, boxes=boxes, pool=pool,
                local_max=local_max, guard_candidates=guard_candidates,
                guard_points=np.flatnonzero(guard_candidates.any(1)).astype(np.int64),
                positive_packed=np.packbits(positive.reshape(3072, -1), axis=1, bitorder="little"),
                high_packed=np.packbits(high.reshape(3072, -1), axis=1, bitorder="little"),
                low_packed=np.packbits(low.reshape(3072, -1), axis=1, bitorder="little"),
                valid_sample_map=valid, local_patch=local_patch)


@torch.inference_mode()
def gpu_summary(logits, local_patch, input_hw):
    """Required production statistics; optional pixel archival is elsewhere."""
    high, low = (logits > 1).sum((-1,-2)), (logits > -1).sum((-1,-2))
    quality = high.to(torch.float32) / low.to(torch.float32)
    valid = torch.as_tensor(valid_geometry(input_hw), device=logits.device)
    foreground = (logits > 0) & valid[None,None]
    counts = foreground.sum((-1,-2)).to(torch.int32)
    flat = foreground.flatten(0,1)
    cols, rows = flat.any(-2), flat.any(-1)
    axis = torch.arange(64, device=logits.device, dtype=torch.float32)*4
    minimum_x = torch.where(cols, axis[None], float("inf")).min(-1).values
    minimum_y = torch.where(rows, axis[None], float("inf")).min(-1).values
    maximum_x = torch.where(cols, axis[None], float("-inf")).max(-1).values
    maximum_y = torch.where(rows, axis[None], float("-inf")).max(-1).values
    boxes = torch.stack((minimum_x, minimum_y, maximum_x, maximum_y),1)
    boxes[counts.flatten()==0] = 0
    return dict(quality=quality, valid_count=counts, boxes=boxes,
                intersection=high.to(torch.int32), union=low.to(torch.int32),
                local_max=local_patch.amax((-1,-2)), local_patch=local_patch)


def evidence_from_summaries(parts, input_hw):
    evidence = {name:np.concatenate([p[name] for p in parts],axis=0) for name in parts[0]}
    quality, counts, local_max = evidence["quality"], evidence["valid_count"], evidence["local_max"]
    guards = (local_max>np.float32(0)) & (counts<=16)
    evidence.update(pool=(counts.reshape(-1)>0)&np.isfinite(quality.reshape(-1))&(quality.reshape(-1)>np.float32(.80)),
                    guard_candidates=guards, guard_points=np.flatnonzero(guards.any(1)).astype(np.int64),
                    valid_sample_map=valid_geometry(input_hw))
    return evidence


def stable_nms(indices, boxes, quality, threshold=np.float32(.7)):
    """Descending float32 quality; exact ties keep original point/token index.

    Native max-minus-min bbox convention. Zero area / zero union gives IoU 0.
    """
    indices = np.asarray(indices, dtype=np.int64)
    order = indices[np.lexsort((indices, -quality[indices]))]
    keep = []
    while len(order):
        first = int(order[0]); keep.append(first); rest = order[1:]
        if not len(rest): break
        a, b = boxes[first], boxes[rest]
        wh = np.maximum(np.minimum(a[2:], b[:,2:]) - np.maximum(a[:2], b[:,:2]), np.float32(0))
        intersection = wh[:,0] * wh[:,1]
        area_a = np.prod(np.maximum(a[2:]-a[:2], np.float32(0)))
        whb = np.maximum(b[:,2:]-b[:,:2], np.float32(0))
        union = area_a + whb[:,0]*whb[:,1] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union>0)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def select(evidence, waves):
    assert waves in (2, 4)
    quality = evidence["quality"].reshape(-1)
    remaining = np.flatnonzero(evidence["pool"])
    wave_points, wave_candidates = [], []
    for _ in range(waves):
        kept = stable_nms(remaining, evidence["boxes"], quality)
        points = np.unique(kept // 3)
        wave_candidates.append(kept.tolist()); wave_points.append(points.tolist())
        remaining = remaining[~np.isin(remaining // 3, points)]
    nms_points = np.unique(np.concatenate([np.asarray(p,dtype=np.int64) for p in wave_points]))
    guards = evidence["guard_points"]
    selected = np.union1d(nms_points, guards).astype(np.int64)
    return selected, dict(waves=waves, wave_candidate_indices=wave_candidates,
                         wave_source_points=wave_points, nms_source_points=nms_points.tolist(),
                         guard_source_points=guards.tolist(), guard_only_source_points=np.setdiff1d(guards,nms_points).tolist(),
                         selected_points_original_order=selected.tolist(), preview_pool_candidates=int(evidence["pool"].sum()),
                         guard_count=int(len(guards)), guard_only_count=int(len(np.setdiff1d(guards,nms_points))),
                         absent_guard_candidate_count=int(((evidence["valid_count"]==0)&evidence["guard_candidates"]).sum()),
                         selection_is_approximate=True, AR_guarantee=False)
