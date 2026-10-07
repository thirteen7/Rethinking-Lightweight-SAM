# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
import copy
import json
import numpy as np
import torch
from torch.nn import functional as F
from scripts.flip_study.box_decoder_current_anchor import choose_anchor
from scripts.flip_study.box_region_evidence import RegionSelector, pair_prediction, training_loss
from scripts.flip_study.eval_box_augmented_feedback import input_boxes
from scripts.flip_study.eval_candidate_onpolicy import chunk_inputs
from scripts.flip_study.eval_legacy_box_versions import original_next_point
from scripts.flip_study.light_sam_adapter import CHECKPOINTS, load_sam
from scripts.flip_study.pack_sa1b_subset import sha256
from scripts.flip_study.train_click_residual import PilotDataset, save_torch
from scripts.flip_study.train_feedback_adaptation import candidate_scores
from scripts.side_coco_eval.backbone_edit_model import FORMAT, decode_independent

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def rows_for(args, split):
    rows = json.loads(args.splits.read_text(encoding='utf-8'))['splits'][split]
    limit = {'fit': 256, 'dev': 64, 'test': 32}[split]
    return rows[:1] if args.smoke else rows[:limit]

def prepare(args):
    from scripts.side_coco_eval.prompt_contract import prompt_contract
    if not CHECKPOINTS[args.model].is_file():
        raise FileNotFoundError(f'Download official {args.model} weights: {CHECKPOINTS[args.model]}')
    source = json.loads(args.splits.read_text(encoding='utf-8'))['splits']
    if set(source) != {'fit', 'dev', 'test'}:
        raise ValueError('Expected fit/dev/test SA-1B splits')
    ids = [{row['image_id'] for row in source[key]} for key in ('fit', 'dev', 'test')]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError('SA-1B image splits overlap')
    folder = args.output / args.model
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(format=FORMAT, model=args.model, smoke=args.smoke, source_checkpoint=str(CHECKPOINTS[args.model]), source_sha256=sha256(CHECKPOINTS[args.model]), splits_sha256=sha256(args.splits), split_counts={key: len(source[key]) for key in source}, split_objects={key: sum((len(row['annotation_ids']) for row in source[key])) for key in source}, stage_images=dict(point_fit=1500, box_fit=2900, base_dev=300, base_test=300, late_fit=256, late_dev=64, late_test=32), annotation_selection='exact selected_indices and annotation_ids from split manifest', stages=['base', 'decoder', 'comparator'], prompt_contract=prompt_contract(), frozen=['image_encoder', 'prompt_encoder', 'original_mask_decoder'], decoder='independent full mask decoder, native-majority mask supervision', comparator='current selected versus four new candidates; edit A/R pooled evidence', score='detector_score unchanged; no added quality head', no_external_training=True)
    path = folder / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != spec:
            raise ValueError(f'Training protocol changed: {path}')
    else:
        write_json(path, spec)
    return folder

def decoder_loss(low, raw, target, valid):
    weight = valid.expand_as(low)
    truth = target.expand_as(low)
    bce = (F.binary_cross_entropy_with_logits(low, truth, reduction='none') * weight).sum((-1, -2)) / weight.sum((-1, -2)).clamp_min(1)
    prob = low.sigmoid() * weight
    dice = 1 - (2 * (prob * truth).sum((-1, -2)) + 1) / (prob.sum((-1, -2)) + (truth * weight).sum((-1, -2)) + 1)
    with torch.no_grad():
        binary = (low > 0) & (weight > 0.5)
        actual = (truth > 0.5) & (weight > 0.5)
        iou = (binary & actual).sum((-1, -2)).float() / (binary | actual).sum((-1, -2)).clamp_min(1)
    per = bce + dice
    return per[:, 0].mean() + per[:, 1:].min(1).values.mean() + 0.1 * F.mse_loss(raw, iou)

def decoder_epoch(args, sam, decoder, rows, optimizer=None):
    dataset = PilotDataset(rows)
    order = np.random.default_rng(args.seed + (0 if optimizer else 1000)).permutation(len(rows))
    (losses, steps) = (0.0, 0)
    for (count, index) in enumerate(order, 1):
        n = len(dataset.data[int(index)][1])
        embedding = source_image = source_hw = source_nh = None
        for start in range(0, n, args.prompt_batch):
            end = min(n, start + args.prompt_batch)
            (image, hw, nh, low_gt, valid, legacy, native, _, _, _) = chunk_inputs(dataset, int(index), start, end)
            records = dataset.data[int(index)][1][start:end]
            boxes = input_boxes(records, nh, hw, 'cuda')
            with torch.no_grad():
                if embedding is None:
                    embedding = sam.image_encoder(image)
                    (source_image, source_hw, source_nh) = (image, hw, nh)
                elif hw != source_hw or nh != source_nh or (not torch.equal(image, source_image)):
                    raise RuntimeError(f'Image changed across annotation chunks: {index}')
                native_grid = F.interpolate(native[:, None].float(), hw, mode='area')
                native_grid = F.pad(native_grid, (0, 1024 - hw[1], 0, 1024 - hw[0]))
                native_low = F.interpolate(native_grid, (256, 256), mode='area')
                target = 0.25 * low_gt + 0.75 * native_low
            clicks = boxes.new_empty((len(boxes), 0, 2))
            labels = boxes.new_empty((len(boxes), 0))
            for rnd in range(3):
                (low, raw, _) = decode_independent(sam, decoder, sam.prompt_encoder.get_dense_pe(), args.model, embedding, boxes, clicks, labels)
                loss = decoder_loss(low, raw, target, valid)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    optimizer.step()
                losses += float(loss.detach())
                steps += 1
                if rnd < 2:
                    with torch.no_grad():
                        (xy, lab) = original_next_point(low[:, :1].detach(), low_gt, valid)
                    clicks = torch.cat([clicks, xy], 1)
                    labels = torch.cat([labels, lab], 1)
        if count == 1 or count % 8 == 0 or count == len(rows):
            print(f"{args.model} decoder {('fit' if optimizer else 'dev')} {count}/{len(rows)}", flush=True)
    return losses / steps

def train_decoder(args, folder):
    path = folder / 'independent_decoder.pth'
    if path.is_file():
        return
    sam = load_sam(args.model)
    decoder = copy.deepcopy(sam.mask_decoder).cuda().train().requires_grad_(True)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.decoder_lr, weight_decay=0.0001)
    best = float('inf')
    history = []
    for epoch in range(1, args.decoder_epochs + 1):
        decoder.train()
        fit_loss = decoder_epoch(args, sam, decoder, rows_for(args, 'fit'), optimizer)
        decoder.eval()
        with torch.no_grad():
            dev_loss = decoder_epoch(args, sam, decoder, rows_for(args, 'dev'))
        history.append(dict(epoch=epoch, fit_loss=fit_loss, dev_loss=dev_loss))
        print(f'{args.model} decoder epoch {epoch}: fit={fit_loss:.5f}, dev={dev_loss:.5f}', flush=True)
        if dev_loss < best:
            best = dev_loss
            state = {key: value.detach().cpu().clone() for (key, value) in decoder.state_dict().items()}
            save_torch(path, dict(format=FORMAT, model=args.model, stage='decoder', state_dict=state, epoch=epoch, best_dev_loss=best, source_sha256=sha256(CHECKPOINTS[args.model]), protocol_sha256=sha256(folder / 'protocol.json'), smoke=args.smoke))
    write_json(folder / 'decoder_training.json', dict(complete=True, history=history, best_dev_loss=best, checkpoint_sha256=sha256(path), smoke=args.smoke))

@torch.inference_mode()
def collect_comparator(args, folder, model, head, stage, split):
    rows = rows_for(args, split)
    dataset = PilotDataset(rows)
    cache = folder / 'comparator_cache' / f'stage{stage}' / split
    cache.mkdir(parents=True, exist_ok=True)
    for (number, row) in enumerate(rows, 1):
        path = cache / f"{row['image_id']:012d}.pth"
        if path.is_file():
            continue
        pieces = {key: [] for key in ('tokens', 'stats', 'geometry', 'compatible', 'iou', 'native_iou')}
        embedding = source_image = source_hw = source_nh = None
        for start in range(0, len(row['annotation_ids']), args.prompt_batch):
            end = min(len(row['annotation_ids']), start + args.prompt_batch)
            (image, hw, nh, low_gt, valid, legacy, native, _, _, _) = chunk_inputs(dataset, number - 1, start, end)
            boxes = input_boxes(dataset.data[number - 1][1][start:end], nh, hw, 'cuda')
            if embedding is None:
                embedding = model.sam.image_encoder(image)
                (source_image, source_hw, source_nh) = (image, hw, nh)
            elif hw != source_hw or nh != source_nh or (not torch.equal(image, source_image)):
                raise RuntimeError(f"Image changed across annotation chunks: {row['image_id']}")
            clicks = boxes.new_empty((len(boxes), 0, 2))
            labels = boxes.new_empty((len(boxes), 0))
            for rnd in range(3):
                (lows, features, _, _) = model.box_candidates(embedding, boxes, clicks, labels, hw)
                (tokens, stats, geometry, compatible) = features
                (old_iou, native_iou) = candidate_scores(model.sam, lows, hw, nh, legacy, native)
                values = (tokens, stats, geometry, compatible, old_iou, native_iou)
                for (key, value) in zip(pieces, values):
                    pieces[key].append(value.detach().cpu())
                if stage and head is not None:
                    choice = choose_anchor(pair_prediction(head, tokens, stats, geometry), compatible)
                else:
                    choice = torch.zeros(len(boxes), device=boxes.device, dtype=torch.long)
                selected = lows[torch.arange(len(choice), device=choice.device), choice, None]
                if rnd < 2:
                    (xy, lab) = original_next_point(selected, low_gt, valid)
                    clicks = torch.cat([clicks, xy], 1)
                    labels = torch.cat([labels, lab], 1)
        save_torch(path, {key: torch.cat(values) for (key, values) in pieces.items()})
        if number == 1 or number % 8 == 0 or number == len(rows):
            print(f'{args.model} comparator stage{stage} {split} {number}/{len(rows)}', flush=True)

def load_comparator_cache(args, folder, stage, split):
    base = folder / 'comparator_cache' / f'stage{stage}' / split
    files = [base / f"{row['image_id']:012d}.pth" for row in rows_for(args, split)]
    parts = [torch.load(path, map_location='cpu', weights_only=True) for path in files]
    return {key: torch.cat([part[key] for part in parts]).cuda() for key in parts[0]}

def comparator_epoch(head, data, optimizer=None):
    order = torch.randperm(len(data['tokens']), device='cuda') if optimizer else torch.arange(len(data['tokens']), device='cuda')
    total = 0.0
    for indices in order.split(256):
        pred = pair_prediction(head, data['tokens'][indices].float(), data['stats'][indices].float(), data['geometry'][indices].float())
        loss = training_loss(pred, data['iou'][indices].float(), data['native_iou'][indices].float(), data['compatible'][indices])
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / len(order)

def train_comparator(args, folder):
    path = folder / 'edit_comparator.pth'
    if path.is_file():
        return
    from scripts.side_coco_eval.box_only_training import BoxOnlyTrainingModel
    model = BoxOnlyTrainingModel(args.model, args.output, quality=False, comparator=False, allow_smoke=args.smoke)
    head = RegionSelector().cuda()
    history = []
    for stage in (0, 1):
        head.eval()
        for split in ('fit', 'dev'):
            collect_comparator(args, folder, model, head if stage else None, stage, split)
        fit = load_comparator_cache(args, folder, stage, 'fit')
        dev = load_comparator_cache(args, folder, stage, 'dev')
        with torch.no_grad():
            stats = torch.cat([fit['stats'][:, 0], fit['stats'][:, 1, :1]], 1).float()
            geometry = torch.cat([fit['geometry'][:, 0], fit['geometry'][:, 1, 1:2]], 1).float()
            head.stat_mean.copy_(stats.mean((0, 1)))
            head.stat_std.copy_(stats.std((0, 1), unbiased=False).clamp_min(0.01))
            head.box_mean.copy_(geometry.mean((0, 1)))
            head.box_std.copy_(geometry.std((0, 1), unbiased=False).clamp_min(0.01))
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.comparator_lr, weight_decay=0.0001)
        best = float('inf')
        for epoch in range(1, args.comparator_epochs + 1):
            head.train()
            fit_loss = comparator_epoch(head, fit, optimizer)
            head.eval()
            with torch.no_grad():
                dev_loss = comparator_epoch(head, dev)
            history.append(dict(stage=stage, epoch=epoch, fit_loss=fit_loss, dev_loss=dev_loss))
            print(f'{args.model} comparator stage{stage} epoch{epoch}: fit={fit_loss:.5f}, dev={dev_loss:.5f}', flush=True)
            if dev_loss < best:
                best = dev_loss
                best_state = {key: value.detach().cpu().clone() for (key, value) in head.state_dict().items()}
        head.load_state_dict(best_state, strict=True)
        save_torch(folder / f'edit_comparator_stage{stage}.pth', dict(format=FORMAT, model=args.model, stage=stage, state_dict=best_state, best_dev_loss=best, decoder_sha256=sha256(folder / 'independent_decoder.pth'), smoke=args.smoke))
        del fit, dev
        torch.cuda.empty_cache()
    save_torch(path, dict(format=FORMAT, model=args.model, stage='comparator', state_dict=best_state, decoder_sha256=sha256(folder / 'independent_decoder.pth'), smoke=args.smoke, complete_training=True))
    write_json(folder / 'comparator_training.json', dict(complete=True, history=history, checkpoint_sha256=sha256(path), smoke=args.smoke))
