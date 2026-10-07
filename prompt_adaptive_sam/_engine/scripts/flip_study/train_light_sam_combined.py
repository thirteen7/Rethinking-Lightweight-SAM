# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.flip_study.candidate_selector import GainSelector, prediction_features
from scripts.flip_study.click_residual import ClickResidualHead, crop, make_inputs
from scripts.flip_study.eval_legacy_box_versions import original_next_point
from scripts.flip_study.feedback_adaptation import TemporalSelector
from scripts.flip_study.infer_light_sam_coco_full import prepare_chunk
from scripts.flip_study.light_sam_adapter import CHECKPOINTS, load_sam
from scripts.flip_study.light_sam_combined import LightFeatureEngine, choose_point, repair_point, score_point
from scripts.flip_study.local_attention_selector import LocalSelector
from scripts.flip_study.pack_sa1b_subset import sha256
from scripts.flip_study.train_click_residual import loss_fn as residual_loss
from scripts.flip_study.train_light_sam_route import DATA, SPLITS, dataset as source_dataset
from scripts.flip_study.train_local_attention import loss_fn as router_loss
ROUTER_LOSS = dict(small_weight=2.0, harm_weight=0.5, harm_positive_weight=2.0, unsafe_positive_gain_weight=0.25)
FLOAT_FIELDS = {'tokens', 'stats', 'context', 'latest_context', 'temporal', 'iou', 'inputs', 'base', 'support', 'target', 'area'}
EXPLICIT_SPLITS = None
EXPLICIT_ROWS = None

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)

def load_torch(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def checkpoint(folder, component, model, state_dict, **extra):
    save_torch(folder / f'{component}.pth', dict(format_version=2, model=model, component=component, state_dict={k: v.detach().cpu() for (k, v) in state_dict.items()}, **extra))

def ids_for(data, split, limit):
    if EXPLICIT_SPLITS is not None:
        lookup = {info['id']: index for (index, (info, _)) in enumerate(data.data)}
        ids = EXPLICIT_SPLITS[split]
        if limit:
            ids = ids[:limit]
        return [(lookup[image_id], image_id) for image_id in ids]
    (start, end) = SPLITS[split]
    return [(index, data.data[index][0]['id']) for index in range(start, min(end, start + limit) if limit else end)]

def dataset():
    if EXPLICIT_ROWS is None:
        return source_dataset()
    from scripts.flip_study import train_click_residual as pilot
    pilot.DATA = DATA
    rows = [row for split in ('fit', 'dev', 'test') for row in EXPLICIT_ROWS[split]]
    data = pilot.PilotDataset(rows)
    for (row, (info, annotations)) in zip(rows, data.data):
        if info['id'] != row['image_id'] or [a['id'] for a in annotations] != row['annotation_ids']:
            raise ValueError(f"Incorrect selected SA-1B annotations: {row['image_id']}")
    return data

def protocol(folder, args, stage, split, image_ids, dependencies=None):
    from scripts.side_coco_eval.prompt_contract import prompt_contract
    spec = dict(format_version=2, model=args.model, stage=stage, split=split, image_ids=image_ids, max_objects=args.max_objects, checkpoint_sha256=sha256(CHECKPOINTS[args.model]), manifest_sha256=sha256(DATA / 'manifest.json'), annotation_selection_sha256=sha256(args.split_json) if args.split_json else None, trainer_sha256=sha256(Path(__file__)), dependency_sha256={name: sha256(path) for (name, path) in (dependencies or {}).items()}, architecture='point attention -> temporal feedback -> local residual; box gain/risk router', prompt_contract=prompt_contract(), decoder_batch=args.decoder_batch, base_prompts=args.base_prompts if stage == 'base' else None)
    path = folder / 'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != spec:
            old = {key: value for (key, value) in previous.items() if key != 'trainer_sha256'}
            new = {key: value for (key, value) in spec.items() if key != 'trainer_sha256'}
            if not (args.resume_frozen_embedding and old == new):
                raise ValueError(f'Collection protocol changed: {path}')
            save_json(folder / 'embedding_resume.json', dict(prior_trainer_sha256=previous['trainer_sha256'], current_trainer_sha256=spec['trainer_sha256'], reason='same frozen image embedding reused across identical annotation chunks', existing_per_image_caches_preserved=True))
            save_json(path, spec)
    else:
        save_json(path, spec)
    return spec

def seed_image(image_id, model, stage):
    salt = sum(((index + 1) * ord(char) for (index, char) in enumerate(model + stage)))
    torch.manual_seed((image_id * 1009 + salt) % 2147483647)

def candidate_iou(sam, low, hw, native_hw, truth):
    scores = []
    for slot in range(4):
        mask = sam.postprocess_masks(low[:, slot:slot + 1], hw, native_hw)[:, 0] > 0
        inter = (mask & truth).flatten(1).sum(1)
        union = (mask | truth).flatten(1).sum(1).clamp_min(1)
        scores.append(inter.float() / union)
    return torch.stack(scores, 1)

def pack(rows):
    result = {}
    for (key, value) in rows.items():
        result[key] = torch.cat(value).cpu()
        if result[key].is_floating_point() and key not in ('iou', 'area'):
            result[key] = result[key].half()
    return result

@torch.inference_mode()
def collect_base(args, data):
    folder = args.output / args.model / 'base'
    sam = load_sam(args.model)
    engine = LightFeatureEngine(sam, args.model)
    for split in ('fit', 'dev'):
        pairs = ids_for(data, split, args.max_images)
        protocol(folder / split, args, 'base', split, [i for (_, i) in pairs])
        for (done, (index, image_id)) in enumerate(pairs, 1):
            path = folder / split / f'{image_id:012d}.pt'
            if path.exists():
                continue
            seed_image(image_id, args.model, 'base')
            n = len(data.data[index][1])
            if args.max_objects:
                n = min(n, args.max_objects)
            point_rows = {k: [] for k in ('tokens', 'stats', 'context', 'valid', 'compatible', 'iou', 'area')}
            box_rows = {k: [] for k in ('tokens', 'stats', 'compatible', 'iou', 'area')}
            embedding = source_image = source_hw = source_nh = None
            for first in range(0, n, args.decoder_batch):
                last = min(first + args.decoder_batch, n)
                (image, hw, nh, low_gt, valid_gt, legacy, point, labels, boxes) = prepare_chunk(data, index, first, last)
                if embedding is None:
                    embedding = sam.image_encoder(image)
                    (source_image, source_hw, source_nh) = (image, hw, nh)
                elif hw != source_hw or nh != source_nh or (not torch.equal(image, source_image)):
                    raise RuntimeError(f'Image changed across annotation chunks: {image_id}')
                if 'point' in args.base_prompts:
                    point_bundle = engine.point_bundle(embedding, image, point, labels, hw)
                    values = dict(tokens=point_bundle['tokens'], stats=point_bundle['stats'], context=point_bundle['context'], valid=point_bundle['valid'], compatible=point_bundle['compatible'], iou=candidate_iou(sam, point_bundle['low'], hw, nh, legacy), area=torch.tensor([a['area'] for a in data.data[index][1][first:last]], device='cuda'))
                    for key in point_rows:
                        point_rows[key].append(values[key])
                coords = torch.empty(last - first, 0, 2, device='cuda')
                click_labels = torch.empty(last - first, 0, device='cuda')
                for round_index in (1, 2, 3):
                    (low, quality, tokens) = engine.decode(embedding, coords if coords.shape[1] else None, click_labels if click_labels.shape[1] else None, boxes)
                    feature_coords = coords if coords.shape[1] else boxes.reshape(last - first, 2, 2).mean(1)[:, None]
                    feature_labels = click_labels if click_labels.shape[1] else torch.full((last - first, 1), -1.0, device='cuda')
                    (stats, compatible) = prediction_features(low, quality, feature_coords, feature_labels, hw, round_index)
                    values = dict(tokens=tokens, stats=stats, compatible=compatible, iou=candidate_iou(sam, low, hw, nh, legacy), area=torch.tensor([a['area'] for a in data.data[index][1][first:last]], device='cuda'))
                    for key in box_rows:
                        box_rows[key].append(values[key])
                    if round_index < 3:
                        (xy, lab) = original_next_point(low[:, :1], low_gt, valid_gt)
                        coords = torch.cat([coords, xy], 1)
                        click_labels = torch.cat([click_labels, lab], 1)
            save_torch(path, dict(model=args.model, image_id=image_id, point=pack(point_rows) if 'point' in args.base_prompts else {}, box=pack(box_rows)))
            if done == 1 or done % (8 if args.model == 'vith' else 25) == 0 or done == len(pairs):
                print(f'{args.model} base {split}: {done}/{len(pairs)}', flush=True)

def load_rows(folder, ids, key):
    rows = []
    for image_id in ids:
        path = folder / f'{image_id:012d}.pt'
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(load_torch(path)[key])
    return {field: torch.cat([row[field] for row in rows]) for field in rows[0]}

def to_device(data, indices, device='cuda'):
    return {k: v[indices].to(device, non_blocking=True).float() if k in FLOAT_FIELDS else v[indices].to(device, non_blocking=True) for (k, v) in data.items()}

def fit_router(args, fit, dev, component, output, initial=None):
    torch.manual_seed(20260929)
    model = LocalSelector('local_attention') if component == 'point_attention' else TemporalSelector() if component.startswith('feedback_') else GainSelector()
    if initial is not None:
        (missing, unexpected) = model.load_state_dict(initial, strict=False)
        if unexpected or any((not key.startswith('temporal_') for key in missing)):
            raise ValueError(f'Incompatible initial weights: {missing}, {unexpected}')
    else:
        model.stat_mean.copy_(fit['stats'].float().mean((0, 1)))
        model.stat_std.copy_(fit['stats'].float().std((0, 1), unbiased=False).clamp_min(1e-05))
        if isinstance(model, LocalSelector):
            (sums, squares) = (torch.zeros(42), torch.zeros(42))
            count = 0
            for start in range(0, len(fit['context']), 256):
                end = start + 256
                context = fit['context'][start:end].float()
                valid = fit['valid'][start:end]
                values = context[valid]
                sums += values.sum(0)
                squares += (values * values).sum(0)
                count += len(values)
            mean = sums / count
            std = (squares / count - mean * mean).clamp_min(1e-10).sqrt()
            model.context_mean.copy_(mean)
            model.context_std.copy_(std)
    if isinstance(model, TemporalSelector) and initial is not None and ('temporal_mean' not in initial):
        model.temporal_mean.copy_(fit['temporal'].float().mean((0, 1)))
        model.temporal_std.copy_(fit['temporal'].float().std((0, 1), unbiased=False).clamp_min(0.01))
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001 if initial is None else 0.0003, weight_decay=0.0001)
    (best, best_state, bad, history) = (float('inf'), None, 0, [])
    epochs = args.epochs if args.epochs else 20 if initial is None else 3
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(fit['tokens']))
        total = 0.0
        for indices in order.split(256):
            batch = to_device(fit, indices)
            prediction = score_point(model, batch) if isinstance(model, LocalSelector) else model(batch['tokens'], batch['stats'])
            target = 100 * (batch['iou'][:, 1:] - batch['iou'][:, :1])
            loss = router_loss(prediction, target, batch['area'], ROUTER_LOSS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        dev_total = 0.0
        with torch.inference_mode():
            for indices in torch.arange(len(dev['tokens'])).split(256):
                batch = to_device(dev, indices)
                prediction = score_point(model, batch) if isinstance(model, LocalSelector) else model(batch['tokens'], batch['stats'])
                target = 100 * (batch['iou'][:, 1:] - batch['iou'][:, :1])
                dev_total += float(router_loss(prediction, target, batch['area'], ROUTER_LOSS)) * len(indices)
        value = dev_total / len(dev['tokens'])
        history.append(dict(epoch=epoch, fit_loss=total / len(fit['tokens']), dev_loss=value))
        print(f"{args.model} {component} epoch={epoch} fit={history[-1]['fit_loss']:.5f} dev={value:.5f}", flush=True)
        if value < best:
            (best, bad) = (value, 0)
            best_state = {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    checkpoint(output, component, args.model, best_state, best_dev_loss=best, history=history)
    return best_state

def fit_base(args):
    folder = args.output / args.model / 'base'
    protocols = {split: json.loads((folder / split / 'protocol.json').read_text()) for split in ('fit', 'dev')}
    destination = args.output / args.model
    if not (destination / 'point_attention.pth').exists():
        point_ids = protocols['fit']['image_ids'][:1500]
        fit = load_rows(folder / 'fit', point_ids, 'point')
        dev = load_rows(folder / 'dev', protocols['dev']['image_ids'], 'point')
        fit_router(args, fit, dev, 'point_attention', destination)
        del fit, dev
    if not (destination / 'box_selector.pth').exists():
        fit = load_rows(folder / 'fit', protocols['fit']['image_ids'], 'box')
        dev = load_rows(folder / 'dev', protocols['dev']['image_ids'], 'box')
        fit_router(args, fit, dev, 'box_selector', destination)

def load_head(path, constructor, model, component):
    state = load_torch(path)
    if (state.get('format_version'), state.get('model'), state.get('component')) != (2, model, component):
        raise ValueError(f'Wrong {component} checkpoint: {path}')
    head = constructor().cuda().eval().requires_grad_(False)
    head.load_state_dict(state['state_dict'], strict=True)
    return head

@torch.inference_mode()
def collect_feedback(args, data, stage):
    destination = args.output / args.model
    dependencies = {'point_attention': destination / 'point_attention.pth'}
    if stage:
        dependencies['feedback_0'] = destination / 'feedback_0.pth'
    first_head = load_head(destination / 'point_attention.pth', lambda : LocalSelector('local_attention'), args.model, 'point_attention')
    stage_head = first_head if stage == 0 else load_head(destination / 'feedback_0.pth', TemporalSelector, args.model, 'feedback_0')
    sam = load_sam(args.model)
    engine = LightFeatureEngine(sam, args.model)
    for split in ('fit', 'dev'):
        limit = min(args.max_images, 256 if split == 'fit' else 64) if args.max_images else 256 if split == 'fit' else 64
        pairs = ids_for(data, split, limit)
        folder = destination / f'feedback_{stage}' / split
        protocol(folder, args, f'feedback_{stage}', split, [i for (_, i) in pairs], dependencies)
        for (done, (index, image_id)) in enumerate(pairs, 1):
            path = folder / f'{image_id:012d}.pt'
            if path.exists():
                continue
            seed_image(image_id, args.model, f'feedback_{stage}')
            n = len(data.data[index][1])
            if args.max_objects:
                n = min(n, args.max_objects)
            rows = {k: [] for k in ('tokens', 'stats', 'context', 'valid', 'latest_context', 'latest_valid', 'temporal', 'compatible', 'iou', 'area')}
            embedding = source_image = source_hw = source_nh = None
            for start in range(0, n, args.decoder_batch):
                end = min(start + args.decoder_batch, n)
                (image, hw, nh, low_gt, valid_gt, legacy, coords, labels, _) = prepare_chunk(data, index, start, end)
                if embedding is None:
                    embedding = sam.image_encoder(image)
                    (source_image, source_hw, source_nh) = (image, hw, nh)
                elif hw != source_hw or nh != source_nh or (not torch.equal(image, source_image)):
                    raise RuntimeError(f'Image changed across annotation chunks: {image_id}')
                first_bundle = engine.point_bundle(embedding, image, coords, labels, hw)
                (previous, _) = choose_point(first_head, first_bundle)
                for _round in (2, 3):
                    (xy, lab) = original_next_point(previous, low_gt, valid_gt)
                    (coords, labels) = (torch.cat([coords, xy], 1), torch.cat([labels, lab], 1))
                    bundle = engine.point_bundle(embedding, image, coords, labels, hw, previous)
                    values = {k: bundle[k] for k in rows if k in bundle}
                    values['iou'] = candidate_iou(sam, bundle['low'], hw, nh, legacy)
                    values['area'] = torch.tensor([a['area'] for a in data.data[index][1][start:end]], device='cuda')
                    for key in rows:
                        rows[key].append(values[key])
                    (previous, _) = choose_point(stage_head, bundle)
            save_torch(path, dict(model=args.model, image_id=image_id, feedback=pack(rows)))
            if done == 1 or done % (8 if args.model == 'vith' else 25) == 0 or done == len(pairs):
                print(f'{args.model} feedback_{stage} {split}: {done}/{len(pairs)}', flush=True)

def fit_feedback(args, stage):
    destination = args.output / args.model
    component = f'feedback_{stage}'
    if (destination / f'{component}.pth').exists():
        return
    folder = destination / component
    paths = {split: json.loads((folder / split / 'protocol.json').read_text())['image_ids'] for split in ('fit', 'dev')}
    fit = load_rows(folder / 'fit', paths['fit'], 'feedback')
    dev = load_rows(folder / 'dev', paths['dev'], 'feedback')
    if stage:
        old = destination / 'feedback_0'
        ids = json.loads((old / 'fit/protocol.json').read_text())['image_ids']
        prior = load_rows(old / 'fit', ids, 'feedback')
        fit = {k: torch.cat([fit[k], prior[k]]) for k in fit}
    source = destination / ('point_attention.pth' if stage == 0 else 'feedback_0.pth')
    initial = load_torch(source)['state_dict']
    fit_router(args, fit, dev, component, destination, initial=initial)
    if stage == 1:
        state = load_torch(destination / 'feedback_1.pth')
        checkpoint(destination, 'point_feedback', args.model, state['state_dict'], source='feedback_1', best_dev_loss=state['best_dev_loss'])

@torch.inference_mode()
def collect_residual(args, data, stage):
    destination = args.output / args.model
    dependencies = {'point_attention': destination / 'point_attention.pth', 'point_feedback': destination / 'point_feedback.pth'}
    if stage:
        dependencies['residual_0'] = destination / 'residual_0.pth'
    first_head = load_head(destination / 'point_attention.pth', lambda : LocalSelector('local_attention'), args.model, 'point_attention')
    feedback_head = load_head(destination / 'point_feedback.pth', TemporalSelector, args.model, 'point_feedback')
    preceding_repair = None if stage == 0 else load_head(destination / 'residual_0.pth', ClickResidualHead, args.model, 'residual_0')
    preceding_scale = 0.0 if stage == 0 else float(load_torch(destination / 'residual_0.pth')['scale'])
    sam = load_sam(args.model)
    engine = LightFeatureEngine(sam, args.model)
    for split in ('fit', 'dev'):
        limit = min(args.max_images, 256 if split == 'fit' else 64) if args.max_images else 256 if split == 'fit' else 64
        pairs = ids_for(data, split, limit)
        folder = destination / f'residual_{stage}' / split
        protocol(folder, args, f'residual_{stage}', split, [i for (_, i) in pairs], dependencies)
        for (done, (index, image_id)) in enumerate(pairs, 1):
            path = folder / f'{image_id:012d}.pt'
            if path.exists():
                continue
            seed_image(image_id, args.model, f'residual_{stage}')
            n = len(data.data[index][1])
            if args.max_objects:
                n = min(n, args.max_objects)
            rows = {k: [] for k in ('inputs', 'base', 'support', 'target', 'area')}
            embedding = source_image = source_hw = source_nh = None
            for start in range(0, n, args.decoder_batch):
                end = min(start + args.decoder_batch, n)
                (image, hw, nh, low_gt, valid_gt, _legacy, coords, labels, _) = prepare_chunk(data, index, start, end)
                if embedding is None:
                    embedding = sam.image_encoder(image)
                    (source_image, source_hw, source_nh) = (image, hw, nh)
                elif hw != source_hw or nh != source_nh or (not torch.equal(image, source_image)):
                    raise RuntimeError(f'Image changed across annotation chunks: {image_id}')
                first_bundle = engine.point_bundle(embedding, image, coords, labels, hw)
                (previous, _) = choose_point(first_head, first_bundle)
                for _round in (2, 3):
                    (xy, lab) = original_next_point(previous, low_gt, valid_gt)
                    (coords, labels) = (torch.cat([coords, xy], 1), torch.cat([labels, lab], 1))
                    bundle = engine.point_bundle(embedding, image, coords, labels, hw, previous, repair=True)
                    (selected, _) = choose_point(feedback_head, bundle)
                    (inputs, base, support) = make_inputs(bundle['patch_features'], image, selected, previous, coords, labels, hw, bundle['region'])
                    values = dict(inputs=inputs, base=base, support=support, target=crop((low_gt > 0).float(), bundle['region']), area=torch.tensor([a['area'] for a in data.data[index][1][start:end]], device='cuda'))
                    for key in rows:
                        rows[key].append(values[key])
                    previous = repair_point(preceding_repair, bundle, image, coords, labels, hw, previous, selected, preceding_scale)
            save_torch(path, dict(model=args.model, image_id=image_id, residual=pack(rows)))
            if done == 1 or done % (8 if args.model == 'vith' else 25) == 0 or done == len(pairs):
                print(f'{args.model} residual_{stage} {split}: {done}/{len(pairs)}', flush=True)

def residual_epoch(head, folders, device='cuda', optimizer=None):
    (total, count) = (0.0, 0)
    for folder in folders:
        ids = json.loads((folder / 'protocol.json').read_text())['image_ids']
        order = torch.randperm(len(ids)).tolist() if optimizer is not None else list(range(len(ids)))
        for i in order:
            row = load_torch(folder / f'{ids[i]:012d}.pt')['residual']
            indices = torch.randperm(len(row['base'])) if optimizer is not None else torch.arange(len(row['base']))
            for selected in indices.split(32):
                batch = to_device(row, selected, device)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    (loss, _) = residual_loss(head, batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                else:
                    with torch.inference_mode():
                        (loss, _) = residual_loss(head, batch)
                total += float(loss.detach()) * len(selected)
                count += len(selected)
    return total / max(count, 1)

@torch.inference_mode()
def calibrate_residual(head, folder):
    scales = (0.0, 0.25, 0.5, 1.0)
    totals = {scale: 0.0 for scale in scales}
    pixels = 0.0
    ids = json.loads((folder / 'protocol.json').read_text())['image_ids']
    head.eval()
    for image_id in ids:
        row = load_torch(folder / f'{image_id:012d}.pt')['residual']
        for indices in torch.arange(len(row['base'])).split(32):
            batch = to_device(row, indices)
            delta = head(batch['inputs'], batch['support'])
            weight = (batch['support'] > 0).float()
            pixels += float(weight.sum())
            for scale in scales:
                error = F.binary_cross_entropy_with_logits(batch['base'] + scale * delta, batch['target'], reduction='none')
                totals[scale] += float((error * weight).sum())
    if not pixels:
        return (0.0, {str(scale): 0.0 for scale in scales})
    losses = {scale: totals[scale] / pixels for scale in scales}
    best = min(scales, key=lambda scale: (losses[scale], scale))
    if losses[best] >= losses[0.0]:
        best = 0.0
    return (best, {str(scale): losses[scale] for scale in scales})

def fit_residual(args, stage):
    destination = args.output / args.model
    component = f'residual_{stage}'
    if (destination / f'{component}.pth').exists():
        return
    folder = destination / component
    torch.manual_seed(20260929 + stage)
    head = ClickResidualHead().cuda()
    if stage:
        head.load_state_dict(load_torch(destination / 'residual_0.pth')['state_dict'], strict=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
    (best, best_state, history) = (float('inf'), None, [])
    epochs = args.epochs if args.epochs else 3
    for epoch in range(1, epochs + 1):
        head.train()
        fit_loss = residual_epoch(head, [folder / 'fit'], optimizer=optimizer)
        head.eval()
        dev_loss = residual_epoch(head, [folder / 'dev'])
        history.append(dict(epoch=epoch, fit_loss=fit_loss, dev_loss=dev_loss))
        print(f'{args.model} {component} epoch={epoch} fit={fit_loss:.5f} dev={dev_loss:.5f}', flush=True)
        if dev_loss < best:
            best = dev_loss
            best_state = {k: v.detach().cpu().clone() for (k, v) in head.state_dict().items()}
    head.load_state_dict(best_state, strict=True)
    (scale, calibration) = calibrate_residual(head, folder / 'dev')
    checkpoint(destination, component, args.model, best_state, best_dev_loss=best, history=history, scale=scale, dev_scale_bce=calibration)
    if stage == 1:
        checkpoint(destination, 'point_residual', args.model, best_state, source='residual_1', best_dev_loss=best, scale=scale, dev_scale_bce=calibration)

def all_phases(args):
    data = dataset()
    collect_base(args, data)
    fit_base(args)
    for stage in (0, 1):
        collect_feedback(args, data, stage)
        fit_feedback(args, stage)
    for stage in (0, 1):
        collect_residual(args, data, stage)
        fit_residual(args, stage)
    destination = args.output / args.model
    save_json(destination / 'complete.json', dict(format_version=2, model=args.model, complete=True, components={name: sha256(destination / f'{name}.pth') for name in ('point_attention', 'point_feedback', 'point_residual', 'box_selector')}, fit_images=len(ids_for(data, 'fit', args.max_images)), point_fit_images=min(1500, len(ids_for(data, 'fit', args.max_images))), box_fit_images=len(ids_for(data, 'fit', args.max_images)), dev_images=len(ids_for(data, 'dev', args.max_images)), fit_objects=sum((min(len(data.data[i][1]), args.max_objects) if args.max_objects else len(data.data[i][1]) for (i, _) in ids_for(data, 'fit', args.max_images))), dev_objects=sum((min(len(data.data[i][1]), args.max_objects) if args.max_objects else len(data.data[i][1]) for (i, _) in ids_for(data, 'dev', args.max_images))), annotation_selection_sha256=sha256(args.split_json) if args.split_json else None, smoke=bool(args.max_images or args.max_objects or args.epochs)))
