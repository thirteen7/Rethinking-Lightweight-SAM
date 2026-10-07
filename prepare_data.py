"""Copy the fixed 3500 SA-1B training subset into the portable train layout."""
import argparse
import json
from pathlib import Path
import shutil
from prompt_adaptive_sam.model import sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='SA-1B root with images/train and annotations/train')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--splits',type=Path,default=Path(__file__).parent/'configs/sa1b_splits.json')
    args=parser.parse_args()
    splits=json.loads(args.splits.read_text())['splits']
    rows=[r for part in ('fit','dev','test') for r in splits[part]]
    if len(rows)!=3500 or len({r['image_id'] for r in rows})!=3500: raise ValueError('Invalid fixed split')
    (args.output/'trainval').mkdir(parents=True,exist_ok=True)
    (args.output/'annotations').mkdir(parents=True,exist_ok=True)
    for row in rows:
        for kind,field,out in [('images','file_name','trainval'),('annotations','annotation_file','annotations')]:
            source=args.source/kind/'train'/row[field]
            destination=args.output/out/row[field]
            expected=row['image_sha256' if kind=='images' else 'annotation_sha256']
            if not destination.is_file():
                if sha256(source)!=expected: raise ValueError('Source differs: '+str(source))
                shutil.copyfile(source,destination)
            if sha256(destination)!=expected: raise ValueError('Prepared file differs: '+str(destination))
    manifest=dict(source='SA-1B',source_split='train',training_images=3500,images=rows)
    (args.output/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n',encoding='utf-8')
    print('Prepared 3500 SA-1B train images with fixed annotation selections')

if __name__=='__main__': main()
