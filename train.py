"""Train the six refinement modules on SA-1B and export one complete model."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['tinysam','mobilesam','vith'],required=True)
    parser.add_argument('--base',type=Path,required=True,help='Official backbone checkpoint')
    parser.add_argument('--data-root',type=Path,required=True,help='Prepared SA-1B train subset')
    parser.add_argument('--output',type=Path,default=Path('runs/train'))
    parser.add_argument('--splits',type=Path,default=HERE/'configs/sa1b_splits.json')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--gpu',default='0',help='CUDA device index or GPU UUID')
    parser.add_argument('--reuse-first',type=Path,help='Reuse the first-click head from a complete released model')
    parser.add_argument('--dry-run',action='store_true',help='Validate inputs and print stages without training')
    args=parser.parse_args()
    if not 1 <= args.batch_size <= 16: parser.error('batch-size must be 1..16')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    from prompt_adaptive_sam._bootstrap import bootstrap
    bootstrap()
    from prompt_adaptive_sam.model import COMPONENTS,sha256,export_components,load_payload,tensor_sha256
    from scripts.flip_study import train_light_sam_combined as base
    from scripts.flip_study import train_click_residual as pilot
    from scripts.flip_study.light_sam_adapter import CHECKPOINTS
    from scripts.side_coco_eval import train_backbone_edit as late
    from scripts.side_coco_eval.prompt_contract import prompt_contract
    if not args.base.is_file(): raise FileNotFoundError(args.base)
    manifest=json.loads((args.data_root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('source_split') != 'train': raise ValueError('Training data must be SA-1B train')
    splits=json.loads(args.splits.read_text(encoding='utf-8'))['splits']
    if {k:len(v) for k,v in splits.items()} != {'fit':2900,'dev':300,'test':300}:
        raise ValueError('Expected disjoint 2900/300/300 SA-1B train split')
    ids=[row['image_id'] for rows in splits.values() for row in rows]
    if len(ids)!=len(set(ids)): raise ValueError('Training split image IDs overlap')
    for rows in splits.values():
        for row in rows:
            image=args.data_root/'trainval'/row['file_name']
            annotation=args.data_root/'annotations'/row['annotation_file']
            if not image.is_file(): raise FileNotFoundError(image)
            if sha256(annotation)!=row['annotation_sha256']: raise ValueError('Annotation differs: '+str(annotation))
            records=json.loads(annotation.read_text())['annotations']
            if [records[i]['id'] for i in row['selected_indices']]!=row['annotation_ids']:
                raise ValueError('Annotation selection differs')
    print(json.dumps(dict(model=args.model,fit='SA-1B train only',base_sha256=sha256(args.base),
        stages=['point selection + box selection','two feedback stages','two residual stages',
                'independent box decoder','two edit-comparator stages','merged export'],
        batch_size=args.batch_size,modules=list(COMPONENTS)),indent=2))
    if args.dry_run: return
    import torch
    import cv2
    if not torch.cuda.is_available(): raise RuntimeError('Training requires CUDA')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.manual_seed(23);torch.cuda.manual_seed_all(23)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.base=args.base.resolve();args.data_root=args.data_root.resolve();args.output=args.output.resolve()
    CHECKPOINTS[args.model]=args.base
    base.DATA=pilot.DATA=args.data_root
    base.EXPLICIT_ROWS=splits
    base.EXPLICIT_SPLITS={key:[row['image_id'] for row in rows] for key,rows in splits.items()}
    common=SimpleNamespace(model=args.model,output=args.output,splits=args.splits.resolve(),
        smoke=False,prompt_batch=args.batch_size,seed=23,decoder_epochs=1,decoder_lr=1e-5,
        comparator_epochs=3,comparator_lr=.00015)
    folder=late.prepare(common)
    if args.reuse_first:
        initial=load_payload(args.reuse_first)
        if initial['model']!=args.model: raise ValueError('First-click head belongs to another backbone')
        state=initial['state_dicts']['point_attention']
        point_path=folder/'point_attention.pth'
        if point_path.is_file():
            previous=torch.load(point_path,map_location='cpu',weights_only=True)
            if tensor_sha256(previous['state_dict'])!=tensor_sha256(state): raise ValueError('First-click head differs')
        else:
            base.checkpoint(folder,'point_attention',args.model,state,reused=True,
                source_checkpoint_sha256=sha256(args.reuse_first))
        del initial
    light_args=SimpleNamespace(model=args.model,output=args.output,split_json=args.splits.resolve(),
        resume_frozen_embedding=False,max_images=0,max_objects=0,epochs=0,
        decoder_batch=args.batch_size,base_prompts=['box'] if args.reuse_first else ['point','box'])
    base.all_phases(light_args)
    late.train_decoder(common,folder)
    late.train_comparator(common,folder)
    late.write_json(folder/'edit_complete.json',dict(complete=True,smoke=False,model=args.model,
        prompt_contract=prompt_contract(),base_sha256=sha256(args.base),splits_sha256=sha256(args.splits),
        components={name:sha256(folder/(name+'.pth')) for name in COMPONENTS}))
    destination=args.output/(args.model+'_prompt_adaptive_v1.pth')
    print('Complete model:',destination,export_components(args.model,args.base,args.output,destination))

if __name__=='__main__': main()
