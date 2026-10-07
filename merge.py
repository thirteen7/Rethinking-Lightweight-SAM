"""Export completed training modules as one checkpoint for one backbone."""
import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['tinysam','mobilesam','vith'],required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--components',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from prompt_adaptive_sam.model import export_components
    digest=export_components(args.model,args.base,args.components,args.output)
    print(f'{args.output}: SHA256 {digest}')

if __name__=='__main__': main()
