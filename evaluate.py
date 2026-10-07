"""Three-round point/box legacy mIoU on complete COCO, LVIS or SA-1B validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent


def prepare(args,evaluator,datasets):
    dataset,indices=datasets.make_dataset(args,masks=args.task=='run')
    ids=[evaluator.identity(args.dataset,dataset,i) for i in indices]
    if not ids or len(ids)!=len(set(ids)): raise ValueError('Empty/duplicate validation IDs')
    if args.dataset=='sa1b':
        used=json.loads(args.splits.read_text())['splits']
        training={row['image_id'] for rows in used.values() for row in rows}
        if training & set(ids): raise ValueError('SA-1B validation overlaps training')
        digest=hashlib.sha256()
        for i in indices:
            image,annotation=map(Path,dataset.data[i])
            if not image.is_file(): raise FileNotFoundError(image)
            digest.update(annotation.name.encode()+b'\0'+bytes.fromhex(evaluator.sha(annotation)))
        annotation_sha=digest.hexdigest()
    else:
        source=args.data_root/'annotations'/('instances_val2017.json' if args.dataset=='coco' else 'lvis_v1_val.json')
        annotation_sha=evaluator.sha(source)
        for i in indices: datasets.find_image(args.data_root,dataset.keys[i])
    count=sum(len(datasets.annotation_ids(args.dataset,dataset,i)) for i in indices)
    plan=dict(format='prompt_adaptive_evaluation_v1',dataset=args.dataset,image_ids=ids,
        available_annotated_images=len(dataset),selected_images=len(ids),objects=count,
        subset=args.max_images is not None,annotation_sha256=annotation_sha,
        gt='1024-resized/padded mask, bilinear native restore, threshold >0',
        prompts=['point','box'],rounds=3,initial_point='GT mask centroid',correction='original_next_point on own logits',
        click_seed=0,chunk_size=64,quality_scoring=False,
        sa1b_selection='official index-seeded randperm cap64 before decode' if args.dataset=='sa1b' else 'all noncrowd annotations',
        code_sha256=evaluator.sha(Path(__file__)))
    path=args.output/'plan.json'
    if path.exists() and evaluator.read(path)!=plan: raise ValueError('Plan changed; use a new output directory')
    evaluator.write(path,plan)
    print(f'{args.dataset}: {len(ids)} valid images, {count} targets; point/box x3; chunk64')
    return dataset,indices,plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['tinysam','mobilesam','vith'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',choices=['coco','lvis','sa1b'],required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('runs/evaluate'))
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--max-images',type=int,help='Optional smoke subset, marked explicitly')
    parser.add_argument('--splits',type=Path,default=HERE/'configs/sa1b_splits.json')
    parser.add_argument('--dry-run',action='store_true',help='Check image/annotation coverage without inference')
    cli=parser.parse_args()
    if cli.max_images is not None and cli.max_images<1: parser.error('max-images must be positive')
    if not cli.checkpoint.is_file(): raise FileNotFoundError(cli.checkpoint)
    os.environ['CUDA_VISIBLE_DEVICES']=cli.gpu
    from prompt_adaptive_sam import datasets
    from prompt_adaptive_sam.model import load_model,load_payload,sha256
    from scripts.side_coco_eval import eval_backbone_interactive as evaluator
    import torch
    import numpy as np
    import cv2
    args=SimpleNamespace(**vars(cli),task='prepare' if cli.dry_run else 'run',
        chunk_size=64,point_from='mask-center',point_seed=0,prompt_types=['point','box'],log_every=1,
        model_root=cli.checkpoint.parent)
    dataset,indices,plan=prepare(args,evaluator,datasets)
    if cli.dry_run: return
    torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('Validation inference requires CUDA')
    payload=load_payload(args.checkpoint)
    if payload['model']!=args.model: raise ValueError('Checkpoint belongs to another model')
    del payload
    protocol=dict(model=args.model,checkpoint_sha256=sha256(args.checkpoint),plan_sha256=evaluator.sha(args.output/'plan.json'),
        evaluator_sha256=evaluator.sha(Path(evaluator.__file__)),torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,python_version=platform.python_version(),
        device=torch.cuda.get_device_name(0),prompt_contract='minus_two_zero_v1',added_quality_head=False)
    path=args.output/'run_protocol.json'
    if path.exists() and evaluator.read(path)!=protocol: raise ValueError('Weights/runtime changed; use a new output')
    evaluator.write(path,protocol)
    evaluator.model_protocol=lambda _args,_plan:evaluator.sha(path)
    evaluator.BackboneEditModel=lambda name,_root,quality=False:load_model(args.checkpoint,'cuda')
    evaluator.annotation_ids=datasets.annotation_ids
    if args.dataset=='sa1b':
        from qa_sam.utils.common import get_centroid_from_mask
        evaluator.get_centroid_from_mask=get_centroid_from_mask
    original_iou=evaluator.iou
    def checked_iou(pred,truth):
        values=original_iou(pred,truth)
        if not torch.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
            raise ValueError('Illegal IoU; preserve outputs and investigate without clamping')
        return values
    evaluator.iou=checked_iou
    evaluator.run(args,dataset,indices,plan)
    # CPU coverage/range/order acceptance; masks are not retained or recomputed.
    sums={p:np.zeros(3,dtype=np.float64) for p in plan['prompts']}
    total=0
    for i,image_id in zip(indices,plan['image_ids']):
        row=evaluator.read(args.output/'images'/f'{image_id:012d}.json')
        expected=datasets.annotation_ids(args.dataset,dataset,i)
        if row['annotation_ids']!=expected: raise ValueError('GT order mismatch')
        if row['protocol_sha256']!=evaluator.sha(path): raise ValueError('Protocol mismatch')
        total+=len(expected)
        for prompt in sums:
            values=np.asarray(row['prompts'][prompt]['iou'],dtype=np.float64)
            if values.shape!=(3,len(expected)) or not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
                raise ValueError(f'Illegal stored IoU: {image_id}/{prompt}')
            sums[prompt]+=values.sum(axis=1)
    summary=evaluator.read(args.output/'summary.json')
    if total!=plan['objects'] or summary['objects']!=total: raise ValueError('Incomplete targets')
    for prompt in sums:
        means=100*sums[prompt]/total
        if not np.array_equal(means,np.asarray(summary['metrics'][prompt]['miou_percent'])):
            raise ValueError('Summary reaggregation differs')
    evaluator.write(args.output/'verified.json',dict(passed=True,images=len(indices),objects=total,
        checkpoint_sha256=protocol['checkpoint_sha256'],summary_sha256=evaluator.sha(args.output/'summary.json'),
        scope='GT order, stored IoU range and CPU reaggregation; masks not stored'))
    print(json.dumps(summary['metrics'],indent=2))

if __name__=='__main__': main()
