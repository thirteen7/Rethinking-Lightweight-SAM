"""Portable validation paths and the official SA-1B cap64 selection."""
import json
from pathlib import Path
import torch
from ._bootstrap import bootstrap
bootstrap()
from data.coco_dataset import COCODataset
from data.official_sa1b_dataset import SA1BDataset


def find_image(root,file_name):
    root=Path(root)
    paths=[root/'trainval'/file_name,root/file_name,root/'val2017'/file_name,
           root/'train2017'/file_name,root/'images'/file_name]
    for path in paths:
        if path.is_file(): return path
    raise FileNotFoundError('Validation image missing: '+str(root/file_name))


def selected_indices(count,index):
    if count <= 64: return list(range(count))
    generator=torch.Generator(device='cpu').manual_seed(index)
    return torch.randperm(count,generator=generator)[:64].tolist()


class BaselineCapDataset:
    def __init__(self,source):
        self.source=source;self.data=source.data
    def __len__(self): return len(self.source)
    def __getitem__(self,key):
        index,start,end=key
        records=json.loads(Path(self.data[index][1]).read_text())['annotations']
        if start!=0 or end!=min(len(records),64): raise ValueError('Official SA-1B cap64 requires chunk64')
        image,annotation=self.source[index]
        if len(annotation['gt_mask'])!=end: raise ValueError('Cap selection differs')
        return image,annotation


def make_dataset(args,*,masks):
    if args.dataset=='sa1b':
        dataset=BaselineCapDataset(SA1BDataset(str(args.data_root),split='val',load_gt_mask=masks,
            max_allowed_prompts=64,fix_seed=True))
        indices=list(range(len(dataset)))
    else:
        dataset=COCODataset(str(args.data_root),split='val',annotation=args.dataset,
            load_gt_mask=masks,max_allowed_prompts=-1,fix_seed=True)
        indices=sorted(range(len(dataset)),key=lambda i:dataset.data[i][0]['id'])
    if args.max_images: indices=indices[:args.max_images]
    return dataset,indices


def annotation_ids(kind,dataset,index):
    if kind=='sa1b':
        rows=json.loads(Path(dataset.data[index][1]).read_text())['annotations']
        return [int(rows[i]['id']) for i in selected_indices(len(rows),index)]
    return [int(row['id']) for row in dataset.data[index][1]]
