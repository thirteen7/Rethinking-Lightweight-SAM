# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from data.coco_dataset import COCODataset
from data.sa1b_dataset import SA1BDataset
from scripts.side_coco_eval.eval_runtime import legacy_centroid_batched as get_centroid_from_mask
from scripts.flip_study.eval_legacy_box_versions import original_next_point
from scripts.side_coco_eval.backbone_edit_model import BackboneEditModel
from scripts.side_coco_eval.progress import format_progress
from scripts.side_coco_eval.point_sampling import random_foreground_points, SAMPLER_VERSION

def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda : source.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()

def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def evaluator_sha(output):
    return sha(Path(__file__))

def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def identity(dataset_name: str, dataset, index: int) -> int:
    if dataset_name == 'sa1b':
        return int(Path(dataset.data[index][0]).stem.split('_')[-1])
    return int(dataset.data[index][0]['id'])

def annotation_ids(dataset_name: str, dataset, index: int) -> list[int]:
    if dataset_name == 'sa1b':
        return [int(row['id']) for row in read(Path(dataset.data[index][1]))['annotations']]
    return [int(row['id']) for row in dataset.data[index][1]]

def make_dataset(args, *, masks: bool):
    if args.dataset == 'sa1b':
        dataset = SA1BDataset(str(args.data_root), split='val', load_gt_mask=masks, max_allowed_prompts=-1, fix_seed=True)
    else:
        dataset = COCODataset(str(args.data_root), split='val', annotation='coco' if args.dataset == 'coco' else 'lvis', load_gt_mask=masks, max_allowed_prompts=-1, fix_seed=True)
    indices = sorted(range(len(dataset)), key=lambda index: identity(args.dataset, dataset, index))
    if args.max_images:
        indices = indices[:args.max_images]
    return (dataset, indices)

def prepare(args):
    prompts = list(getattr(args, 'prompt_types', ('point', 'box')))
    point_from = getattr(args, 'point_from', 'mask-center')
    point_seed = getattr(args, 'point_seed', 0)
    if len(prompts) != len(set(prompts)) or point_seed < 0:
        raise ValueError('prompt-types must be unique and point-seed nonnegative')
    (dataset, indices) = make_dataset(args, masks=args.task == 'run')
    ids = [identity(args.dataset, dataset, index) for index in indices]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate validation image IDs')
    if args.dataset == 'sa1b':
        source = args.data_root / 'annotations/val'
        training = read(args.model_root / 'splits.json')['splits']
        used = {row['image_id'] for part in ('fit', 'dev', 'test') for row in training[part]}
        if used & set(ids):
            raise ValueError('SA-1B val images overlap backbone training images')
        digest = hashlib.sha256()
        for index in indices:
            path = Path(dataset.data[index][1])
            digest.update(path.name.encode() + b'\x00' + bytes.fromhex(sha(path)))
        annotation_digest = digest.hexdigest()
        missing = [Path(dataset.data[i][0]) for i in indices if not Path(dataset.data[i][0]).is_file()]
    else:
        source = args.data_root / ('annotations/instances_val2017.json' if args.dataset == 'coco' else 'annotations/lvis_v1_val.json')
        annotation_digest = sha(source)
        missing = [args.data_root / 'trainval' / dataset.keys[i] for i in indices if not (args.data_root / 'trainval' / dataset.keys[i]).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} validation images missing; first: {missing[0]}')
    plan = dict(format='backbone_interactive_v1', dataset=args.dataset, image_ids=ids, available_annotated_images=len(dataset), selected_images=len(ids), subset=args.max_images is not None, source_annotations=str(source), annotation_sha256=annotation_digest, sa1b_training_splits_sha256=sha(args.model_root / 'splits.json') if args.dataset == 'sa1b' else None, gt='1024-resized/padded mask, bilinear native restore, threshold >0', prompts=prompts, rounds=3, initial_point='GT mask centroid', correction='original_next_point on own logits', click_seed=0, chunk_size=args.chunk_size, centroid='batched legacy distance transform', quality_scoring=False, runtime_sha256=sha(Path(__file__).with_name('eval_runtime.py')), code_sha256=evaluator_sha(args.output))
    if point_from == 'mask-rand':
        plan.update(initial_point='uniform random pixel inside resized GT mask >0', initial_point_seed=point_seed, initial_point_sampler=SAMPLER_VERSION, point_sampling_sha256=sha(Path(__file__).with_name('point_sampling.py')), centroid=None)
    path = args.output / 'plan.json'
    if path.exists() and read(path) != plan:
        if any((args.output / 'images').glob('*.json')):
            raise ValueError(f'Plan changed after inference; use another output directory: {path}')
        write(path, plan)
    if not path.exists():
        write(path, plan)
    print(f'Prepared {args.dataset}: {len(ids)}/{len(dataset)} annotated images', flush=True)
    return (dataset, indices, plan)

def model_protocol(args, plan):
    from scripts.side_coco_eval.prompt_contract import prompt_contract
    folder = args.model_root / args.model
    required = [folder / name for name in ('edit_quality_complete.json', 'complete.json', 'point_attention.pth', 'point_feedback.pth', 'point_residual.pth', 'box_selector.pth', 'independent_decoder.pth', 'edit_comparator.pth')]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Training incomplete: {missing[0]}')
    complete = read(required[0])
    if not complete.get('complete') or complete.get('smoke'):
        raise ValueError('Only complete, non-smoke training checkpoints may be evaluated')
    if complete.get('prompt_contract') != prompt_contract():
        raise ValueError('Training and inference prompt contracts differ')
    value = dict(prompt_contract=prompt_contract(), plan_sha256=sha(args.output / 'plan.json'), model=args.model, complete_sha256=sha(required[0]), component_sha256={path.name: sha(path) for path in required[2:]}, evaluator_sha256=evaluator_sha(args.output))
    path = args.output / 'run_protocol.json'
    if path.exists() and read(path) != value:
        if any((args.output / 'images').glob('*.json')):
            raise ValueError(f'Checkpoint or evaluator changed after inference: {path}')
        write(path, value)
    if not path.exists():
        write(path, value)
    return sha(path)

def save_rng(states):
    return {prompt: {device: bytes(value.tolist()).hex() for (device, value) in pair.items()} for (prompt, pair) in states.items()}

def restore_rng(saved):
    return {prompt: {device: torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8) for (device, value) in pair.items()} for (prompt, pair) in saved.items()}

def iou(predicted, truth):
    intersection = (predicted & truth).flatten(1).sum(1).float()
    union = (predicted | truth).flatten(1).sum(1).clamp_min(1).float()
    return intersection / union

@torch.inference_mode()
def run(args, dataset, indices, plan):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for inference')
    fingerprint = model_protocol(args, plan)
    model = BackboneEditModel(args.model, args.model_root, quality=False)
    prompts = plan['prompts']
    random_initial = getattr(args, 'point_from', 'mask-center') == 'mask-rand'
    print(f"{args.model}/{args.dataset}: prompt batch={args.chunk_size}; prompts={prompts}; initial point={plan['initial_point']}; point seed={getattr(args, 'point_seed', 0)}; quality scoring disabled for IoU", flush=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    states = {prompt: {'cpu': torch.get_rng_state().clone(), 'cuda': torch.cuda.get_rng_state().clone()} for prompt in prompts}
    completed_before = sum(((args.output / 'images' / f'{image_id:012d}.json').is_file() for image_id in plan['image_ids']))
    if completed_before:
        print(f'{args.model}/{args.dataset} IoU: resuming {completed_before} cached images', flush=True)
    started = time.perf_counter()
    processed_now = 0
    for (done, index) in enumerate(indices, 1):
        image_id = identity(args.dataset, dataset, index)
        path = args.output / 'images' / f'{image_id:012d}.json'
        ids = annotation_ids(args.dataset, dataset, index)
        if path.is_file():
            saved = read(path)
            if saved['protocol_sha256'] != fingerprint or saved['annotation_ids'] != ids:
                raise ValueError(f'Incompatible cached image: {path}')
            states = restore_rng(saved['rng_after'])
            continue
        row = dict(image_id=image_id, annotation_ids=ids, protocol_sha256=fingerprint, prompts={prompt: {'iou': [[], [], []], 'chosen': [[], [], []]} for prompt in prompts})
        if random_initial and 'point' in prompts:
            row['initial_points_xy'] = []
        pending = {prompt: {'iou': [[], [], []], 'chosen': [[], [], []]} for prompt in prompts}
        embedding = None
        for start in range(0, len(ids), args.chunk_size):
            end = min(start + args.chunk_size, len(ids))
            (image, ann) = dataset[index, start, end]
            image = image[None].cuda(non_blocking=True)
            hw = tuple(map(int, ann['img_size_before_pad'][1:]))
            nh = (int(ann['info']['height']), int(ann['info']['width']))
            gt = ann['gt_mask'].float().cuda()[:, None]
            truth = F.interpolate(gt[:, :, :hw[0], :hw[1]], nh, mode='bilinear', align_corners=False)[:, 0] > 0
            low_gt = F.interpolate(gt, (256, 256), mode='bilinear', align_corners=False)
            valid = gt.new_zeros((1, 1, 1024, 1024))
            valid[:, :, :hw[0], :hw[1]] = 1
            valid = F.interpolate(valid, (256, 256), mode='bilinear', align_corners=False)
            boxes = ann['prompt_box'].float().cuda()
            point = None
            if 'point' in prompts:
                if random_initial:
                    sampled = random_foreground_points(ann['gt_mask'], dataset=args.dataset, image_id=image_id, annotation_ids=ids[start:end], seed=args.point_seed)
                    row['initial_points_xy'].extend(sampled[:, 0].tolist())
                    point = sampled.float().cuda()
                else:
                    point = get_centroid_from_mask(gt > 0.5).float().cuda()
            if embedding is None:
                embedding = model.sam.image_encoder(image)
            for prompt in prompts:
                torch.set_rng_state(states[prompt]['cpu'])
                torch.cuda.set_rng_state(states[prompt]['cuda'])
                coords = point.clone() if prompt == 'point' else boxes.new_empty((len(boxes), 0, 2))
                labels = point.new_ones((len(boxes), 1)) if prompt == 'point' else boxes.new_empty((len(boxes), 0))
                previous = None
                for round_index in range(3):
                    (mask, chosen, low) = model.predict(embedding, image, coords, labels, hw, nh, prompt, boxes if prompt == 'box' else None, previous)
                    if mask.shape != truth.shape:
                        raise ValueError(f'Mask/GT shape mismatch: {mask.shape}/{truth.shape}')
                    pending[prompt]['iou'][round_index].append(iou(mask, truth))
                    pending[prompt]['chosen'][round_index].append(chosen)
                    model.last_trace = None
                    previous = low
                    if round_index < 2:
                        (xy, lab) = original_next_point(low, low_gt, valid)
                        coords = torch.cat((coords, xy), 1)
                        labels = torch.cat((labels, lab), 1)
                states[prompt] = {'cpu': torch.get_rng_state().clone(), 'cuda': torch.cuda.get_rng_state().clone()}
        for prompt in prompts:
            for round_index in range(3):
                for key in ('iou', 'chosen'):
                    row['prompts'][prompt][key][round_index] = torch.cat(pending[prompt][key][round_index]).cpu().tolist()
        row['rng_after'] = save_rng(states)
        write(path, row)
        processed_now += 1
        if done == 1 or done % args.log_every == 0 or done == len(indices):
            print(format_progress(f'{args.model}/{args.dataset} IoU', done, len(indices), started, processed_now=processed_now), flush=True)
    summarize(args, plan, fingerprint)

def summarize(args, plan, fingerprint=None):
    if fingerprint is None:
        fingerprint = sha(args.output / 'run_protocol.json')
    sums = {prompt: np.zeros(3, dtype=np.float64) for prompt in plan['prompts']}
    counts = {prompt: np.zeros(3, dtype=np.int64) for prompt in sums}
    objects = 0
    points_digest = hashlib.sha256() if 'initial_point_sampler' in plan and 'point' in sums else None
    for image_id in plan['image_ids']:
        row = read(args.output / 'images' / f'{image_id:012d}.json')
        if row['protocol_sha256'] != fingerprint:
            raise ValueError(f'Mismatched image protocol: {image_id}')
        objects += len(row['annotation_ids'])
        if points_digest is not None:
            xy = row['initial_points_xy']
            if len(xy) != len(row['annotation_ids']) or any((len(point) != 2 for point in xy)):
                raise ValueError(f'Invalid initial point record: {image_id}')
            points_digest.update(json.dumps([image_id, row['annotation_ids'], xy], separators=(',', ':')).encode() + b'\n')
        for prompt in sums:
            values = np.asarray(row['prompts'][prompt]['iou'], dtype=float)
            if values.shape != (3, len(row['annotation_ids'])) or not np.isfinite(values).all():
                raise ValueError(f'Invalid IoU row: {image_id}/{prompt}')
            sums[prompt] += values.sum(axis=1)
            counts[prompt] += values.shape[1]
    result = dict(complete=True, model=args.model, dataset=args.dataset, images=len(plan['image_ids']), objects=objects, full_annotated_validation=not plan['subset'], protocol_sha256=fingerprint, gt='legacy resized/restored; native GT not used', metrics={prompt: dict(miou_percent=(100 * sums[prompt] / counts[prompt]).tolist(), objects_per_round=counts[prompt].tolist()) for prompt in sums})
    if points_digest is not None:
        result.update(initial_point=plan['initial_point'], initial_point_seed=plan['initial_point_seed'], initial_points_sha256=points_digest.hexdigest())
    write(args.output / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
