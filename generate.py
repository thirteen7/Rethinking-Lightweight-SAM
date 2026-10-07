"""Generate prompt-free masks from one image using a complete checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from prompt_adaptive_sam import Predictor, EverythingGenerator
from prompt_adaptive_sam.everything import render_instances

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--output',type=Path,default=Path('runs/everything'))
    parser.add_argument('--method',choices=['fsd','dense'],default='fsd')
    parser.add_argument('--grid',type=int,choices=[8,16,32],default=32)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    image = np.asarray(Image.open(args.image).convert('RGB'))
    predictor = Predictor(args.checkpoint,device=args.device)
    masks,info = EverythingGenerator(predictor,points_per_side=args.grid,
        method=args.method,points_per_batch=4 if args.device=='cpu' else 32).generate(image)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'masks.json').write_text(json.dumps(dict(masks=masks,info=info),indent=2)+'\n')
    Image.fromarray(render_instances(image,masks)).save(args.output/'overlay.png')
    print(json.dumps(info,indent=2))
