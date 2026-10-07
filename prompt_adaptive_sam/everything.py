"""Prompt-free mask generation with native SAM or portable FSD-SAM decoding."""
from __future__ import annotations
import colorsys
import time
import numpy as np
import torch
from PIL import Image
from torchvision.ops.boxes import batched_nms
from pycocotools import mask as mask_utils
from qa_sam.utils import amg
from ._everything.bidir_decoder import BidirectionalFactorDecoder
from ._everything.factor_store import FactorStore
from ._everything.selection import gpu_summary, evidence_from_summaries, select


class EverythingGenerator:
    """All prompts are independent positive clicks; no annotation is required.

    FSD-SAM uses the portable factorized decoder, native phase-(0,0) previews,
    four-wave selection and the local small-region guard. Native suffixes
    restore retained requests. This interface does not use the extra
    point/box refinement modules or claim the fused-kernel benchmark timings.
    """
    def __init__(self, predictor, *, points_per_side=16, method='fsd',
                 points_per_batch=32, pred_iou_thresh=.88,
                 stability_thresh=.95, nms_thresh=.7):
        if points_per_side not in (8, 16, 32):
            raise ValueError('Choose an 8, 16, or 32 point grid.')
        if method not in ('fsd', 'dense'):
            raise ValueError('Choose fsd or dense.')
        self.p = predictor
        self.method = method
        self.side = int(points_per_side)
        self.batch = int(points_per_batch)
        if not 1 <= self.batch <= 64:
            raise ValueError('The point batch must be between 1 and 64.')
        self.thresholds = float(pred_iou_thresh), float(stability_thresh), float(nms_thresh)
        self.grid = amg.build_point_grid(self.side)

    def _prompts(self, points):
        p, sam = self.p, self.p.model.sam
        native = points * np.array(p.native_hw[::-1])[None]
        transformed = p.transform.apply_coords(native, p.native_hw)
        coordinates = torch.as_tensor(transformed, device=p.device, dtype=torch.float32)[:, None]
        labels = torch.ones((len(points), 1), device=p.device, dtype=torch.int)
        sparse, dense = sam.prompt_encoder(points=(coordinates, labels), boxes=None, masks=None)
        return dict(image_embeddings=p.embedding, image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense), transformed

    def _filter(self, data, low, head, points):
        p, sam = self.p, self.p.model.sam
        if not torch.isfinite(low).all() or not torch.isfinite(head).all():
            raise RuntimeError('Everything decoding produced non-finite values.')
        masks = sam.postprocess_masks(low, p.input_hw, p.native_hw)
        coords = points * np.array(p.native_hw[::-1])[None]
        part = amg.MaskData(masks=masks.flatten(0, 1), iou_preds=head.flatten(),
                            points=torch.as_tensor(coords.repeat(3, axis=0)))
        part.filter(part['iou_preds'] > self.thresholds[0])
        part['stability_score'] = amg.calculate_stability_score(part['masks'], sam.mask_threshold, 1.)
        part.filter(part['stability_score'] >= self.thresholds[1])
        part['masks'] = part['masks'] > sam.mask_threshold
        part['boxes'] = amg.batched_mask_to_box(part['masks'])
        h, w = p.native_hw
        part.filter(~amg.is_box_near_crop_edge(part['boxes'], [0, 0, w, h], [0, 0, w, h]))
        part['rles'] = amg.mask_to_rle_pytorch(part['masks'])
        del part['masks']
        data.cat(part)

    @torch.inference_mode()
    def generate(self, image=None):
        started = time.perf_counter()
        if image is not None:
            self.p.set_image(image)
        if self.p.embedding is None:
            raise RuntimeError('Supply an image or call Predictor.set_image() first.')
        count, sam = len(self.grid), self.p.model.sam
        selected = np.arange(count, dtype=np.int64)
        engine = store = None
        selection = None
        local_guards = 0
        data = amg.MaskData()
        try:
            if self.method == 'fsd':
                engine = BidirectionalFactorDecoder(sam.mask_decoder)
                dense = sam.prompt_encoder.no_mask_embed.weight.reshape(1, 256, 1, 1).expand(1, 256, 64, 64)
                engine.prepare(self.p.embedding, sam.prompt_encoder.get_dense_pe(), dense)
                store = FactorStore(self.p.device, count=count)
                parts = []
                for start in range(0, count, self.batch):
                    options, transformed = self._prompts(self.grid[start:start+self.batch])
                    state, phase = engine.states(**options)
                    if any(not torch.isfinite(state[key]).all() for key in ('U', 'V', 'scale', 'hyper', 'head')):
                        raise RuntimeError('The factorized decoder failed its finite-state check.')
                    store.put(start, state)
                    preview = engine.phase(phase, state['hyper'], state['hw'])
                    cells = torch.as_tensor(np.floor((transformed[:, ::-1]+.5)/16).astype(np.int64).clip(0, 63), device=self.p.device)
                    local = engine.local_patch(state, cells)
                    summary = gpu_summary(preview, local, self.p.input_hw)
                    parts.append({key: value.cpu().numpy() for key, value in summary.items()})
                    del state, phase, options, preview, local, summary, cells
                evidence = evidence_from_summaries(parts, self.p.input_hw)
                selected, selection = select(evidence, 4)
                local_guards = selection['guard_count']
                del evidence, parts
            for start in range(0, len(selected), self.batch):
                indices = selected[start:start+self.batch]
                points = self.grid[indices]
                if engine is None:
                    options, _ = self._prompts(points)
                    low, head, *unused = sam.mask_decoder.predict_masks(**options)
                    low, head = low[:, 1:4], head[:, 1:4]
                else:
                    state = store.take(torch.as_tensor(indices, device=self.p.device, dtype=torch.long))
                    restored = engine.reconstruct(state)
                    low = engine.complete(restored, state['hyper'])
                    head = state['head']
                    del restored, state
                self._filter(data, low, head, points)
                del low, head
            masks = []
            if len(selected) and len(data['rles']):
                keep = batched_nms(data['boxes'].float(), data['iou_preds'],
                                   torch.zeros_like(data['boxes'][:, 0]), self.thresholds[2])
                data.filter(keep)
                for index, rle in enumerate(data['rles']):
                    encoded = amg.coco_encode_rle(rle)
                    if isinstance(encoded['counts'], bytes):
                        encoded['counts'] = encoded['counts'].decode('ascii')
                    masks.append(dict(segmentation=encoded, area=amg.area_from_rle(rle),
                        bbox=amg.box_xyxy_to_xywh(data['boxes'][index]).cpu().tolist(),
                        predicted_iou=float(data['iou_preds'][index]),
                        stability_score=float(data['stability_score'][index])))
            return masks, dict(method=self.method, backend='portable_pytorch',
                grid=self.side, total_points=count, native_points=len(selected),
                local_guard_points=local_guards, masks=len(masks),
                image_hw=list(self.p.native_hw), point_grid=self.grid.tolist(),
                native_point_indices=selected.tolist(), selection=selection,
                seconds=time.perf_counter()-started, annotation_prompts=False)
        finally:
            if engine is not None:
                engine.clear_cache()


def instance_color(index):
    return tuple(round(channel*255) for channel in colorsys.hsv_to_rgb((.62+index*.61803398875)%1, .58, .93))


def render_instances(image, masks, opacity=.48, selected=None):
    """Render true RLE masks. Small masks remain visible above large ones."""
    result = np.asarray(image, dtype=np.uint8).copy().astype(np.float32)
    for index in sorted(range(len(masks)), key=lambda i: -masks[i]['area']):
        if selected is not None and index != selected:
            continue
        record = masks[index]['segmentation']
        rle = dict(record, counts=record['counts'].encode('ascii'))
        mask = mask_utils.decode(rle).astype(bool)
        if mask.shape != result.shape[:2]:
            raise ValueError('The instance mask geometry differs from the image.')
        result[mask] = result[mask]*(1-opacity)+np.asarray(instance_color(index))*opacity
    return np.clip(result, 0, 255).astype(np.uint8)
