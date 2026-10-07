"""Matched lightweight prompt comparisons and synchronized ViT-H generation."""
from __future__ import annotations
import copy
import hashlib
import time

import numpy as np
import torch

from prompt_adaptive_sam import EverythingGenerator


@torch.inference_mode()
def original_prediction(predictor, name, prompt, stage, points, labels, box, previous):
    """Native public decoder policy, with its own mask feedback."""
    coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    coords = predictor.transform.apply_coords(coords, predictor.native_hw)
    coords = torch.as_tensor(coords, device=predictor.device, dtype=torch.float32)[None]
    labels = torch.as_tensor(labels, device=predictor.device, dtype=torch.float32)[None]
    boxes = None
    if box is not None:
        boxes = predictor.transform.apply_boxes(np.asarray(box, dtype=np.float32).reshape(1, 4), predictor.native_hw)
        boxes = torch.as_tensor(boxes, device=predictor.device, dtype=torch.float32)
    feedback = None if previous is None else torch.as_tensor(previous, device=predictor.device).reshape(1, 1, 256, 256)
    sam = predictor.model.sam
    sparse, dense = sam.prompt_encoder(points=(coords, labels) if coords.shape[1] else None,
                                      boxes=boxes, masks=feedback)
    args = dict(image_embeddings=predictor.embedding, image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
    if name == 'mobilesam':
        args['multimask_output'] = prompt == 'point' and stage == 0
    low, scores = sam.mask_decoder(**args)
    if not torch.isfinite(low).all() or not torch.isfinite(scores).all():
        raise RuntimeError('Non-finite original-model output.')
    choice = int(scores[0].argmax())
    selected = low[:, choice:choice + 1]
    mask = sam.postprocess_masks(selected, predictor.input_hw, predictor.native_hw)[0, 0] > sam.mask_threshold
    return mask.cpu().numpy(), selected[0, 0].cpu().numpy(), choice


def measured_iou(mask, truth):
    mask, truth = np.asarray(mask, dtype=bool), np.asarray(truth, dtype=bool)
    if mask.shape != truth.shape:
        raise ValueError('Prediction and ground truth dimensions differ.')
    intersection, union = int(np.logical_and(mask, truth).sum()), int(np.logical_or(mask, truth).sum())
    if union == 0:
        raise ValueError('The curated target must contain foreground.')
    return dict(intersection=intersection, union=union, iou_percent=100 * intersection / union,
                mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())


@torch.inference_mode()
def compare_prompts(template, name, image, trajectory, truth):
    """Run both lightweight paths for every fixed foreground-only prompt stage.

    Ground truth is used only for reporting IoU. Candidate selection uses model
    scores; the baseline and refinement retain independent previous logits.
    Curated prompt coordinates are fixed in the published example manifest.
    """
    predictor = copy.copy(template)
    predictor.set_image(image)
    baseline_previous = refined_previous = None
    stages = []
    for index, prompt in enumerate(trajectory):
        points = [[p['x'], p['y']] for p in prompt['points']]
        labels = [p['label'] for p in prompt['points']]
        if any(label != 1 for label in labels):
            raise ValueError('The curated demo supports foreground prompts only.')
        box = prompt.get('box')
        mode = 'box' if box is not None else 'point'
        baseline, baseline_previous, base_choice = original_prediction(
            predictor, name, mode, index, points, labels, box, baseline_previous)
        refined, refined_previous, refined_choice = predictor.predict(points, labels, box=box, previous=refined_previous)
        if not np.isfinite(refined_previous).all():
            raise RuntimeError('Non-finite refined output.')
        predictor.model.last_trace = None
        stages.append(dict(points=prompt['points'], box=box,
            original=dict(mask=baseline, candidate=base_choice, **measured_iou(baseline, truth)),
            refined=dict(mask=refined, candidate=refined_choice, **measured_iou(refined, truth))))
    return dict(model=name, shared_prompts=True, independent_feedback=True,
                iou_scope='curated target / legacy ground truth', stages=stages)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def compare_everything(template, image, grid=16, *, run_index=0):
    """Use one frozen ViT-H, one image embedding, and identical SAM filters.

    Both totals include the same measured image encoding cost plus each
    method's mask generation. Loading, warm-up, GPU queue, and rendering are
    outside the measurement. Execution order alternates on repeated runs.
    """
    predictor = copy.copy(template)
    batch = 4 if predictor.device.type == 'cpu' else 32
    predictor.set_image(image)
    for method in ('dense', 'fsd'):
        EverythingGenerator(predictor, method=method, points_per_side=8,
                            points_per_batch=batch).generate()
    synchronize(predictor.device)
    started = time.perf_counter()
    predictor.set_image(image)
    synchronize(predictor.device)
    encoding_ms = (time.perf_counter() - started) * 1000
    order = ('dense', 'fsd') if run_index % 2 == 0 else ('fsd', 'dense')
    results = {}
    for method in order:
        synchronize(predictor.device)
        started = time.perf_counter()
        masks, details = EverythingGenerator(predictor, method=method,
            points_per_side=int(grid), points_per_batch=batch).generate()
        synchronize(predictor.device)
        generation_ms = (time.perf_counter() - started) * 1000
        results[method] = dict(masks=masks, info=details,
            encoding_ms=encoding_ms, generation_ms=generation_ms,
            total_ms=encoding_ms + generation_ms)
    device_name = (torch.cuda.get_device_name(predictor.device)
                   if predictor.device.type == 'cuda' else 'CPU')
    return dict(backbone='SAM ViT-H', same_weights=True, grid=int(grid),
        precision='FP32', device=device_name, order=list(order),
        timing_scope='image encoding + mask generation',
        excluded=['model loading', 'warm-up', 'queue', 'overlay rendering'],
        shared_encoding_counted_in_both=True, results=results)
