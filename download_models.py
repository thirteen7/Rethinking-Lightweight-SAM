"""Download one complete model and verify SHA256; join large transport parts automatically."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen,Request

HERE=Path(__file__).resolve().parent


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''): value.update(block)
    return value.hexdigest()


def download(url,path,expected,size):
    if path.is_file() and path.stat().st_size==size and sha256(path)==expected: return
    temporary=path.with_name(path.name+'.download')
    print('Downloading '+path.name,flush=True)
    request=Request(url,headers={'User-Agent':'Rethinking-Lightweight-SAM/1.0'})
    with urlopen(request,timeout=60) as source,temporary.open('wb') as target:
        shutil.copyfileobj(source,target,8<<20)
    if temporary.stat().st_size!=size or sha256(temporary)!=expected:
        raise ValueError('Download byte count or SHA256 differs: '+path.name)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['tinysam','mobilesam','vith','all'],default='tinysam')
    parser.add_argument('--output',type=Path,default=Path('weights'))
    parser.add_argument('--base-only',action='store_true',help='Also extract the unchanged official base for training')
    args=parser.parse_args()
    index=json.loads((HERE/'models.json').read_text())
    names=list(index['models']) if args.model=='all' else [args.model]
    args.output.mkdir(parents=True,exist_ok=True)
    for name in names:
        model=index['models'][name];destination=args.output/model['filename']
        if not (destination.is_file() and destination.stat().st_size==model['bytes'] and sha256(destination)==model['sha256']):
            parts=[]
            for asset in model['assets']:
                path=args.output/asset['name']
                url=f"https://github.com/{index['repository']}/releases/download/{index['release']}/{asset['name']}"
                download(url,path,asset['sha256'],asset['bytes']);parts.append(path)
            if len(parts)>1:
                temporary=destination.with_suffix('.joining')
                with temporary.open('wb') as output:
                    for part in parts:
                        with part.open('rb') as stream: shutil.copyfileobj(stream,output,8<<20)
                if temporary.stat().st_size!=model['bytes'] or sha256(temporary)!=model['sha256']:
                    raise ValueError('Merged checkpoint SHA256 differs')
                temporary.replace(destination)
                for part in parts: part.unlink()
        print('Verified complete model: '+str(destination),flush=True)
        if args.base_only:
            import torch
            value=torch.load(destination,map_location='cpu',weights_only=True)
            base=args.output/(name+'_official_base.pth')
            torch.save(value['state_dicts']['sam'],base)
            print('Unchanged official base tensors: '+str(base))

if __name__=='__main__': main()
