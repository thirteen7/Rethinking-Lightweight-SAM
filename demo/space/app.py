"""Hosted entry point for the shared research studio."""
import os
from pathlib import Path
import subprocess
import sys
import logging

LOCAL_PROJECT = Path(__file__).resolve().parent.parent.parent
local_checkout = (LOCAL_PROJECT/'demo/gradio_app.py').is_file()
PROJECT = Path(os.environ.get('SAM_DEMO_PROJECT',str(LOCAL_PROJECT) if local_checkout else '/tmp/rethinking-lightweight-sam-paired-v8'))
if local_checkout and not os.environ.get('SPACE_ID'):
    os.environ.setdefault('SAM_DEMO_LOCAL','1')
if not (PROJECT/'demo/gradio_app.py').is_file():
    subprocess.run(['git','clone','--depth','1',
        'https://github.com/thirteen7/Rethinking-Lightweight-SAM.git',str(PROJECT)],check=True)
sys.path.insert(0,str(PROJECT))
from demo.gradio_app import build_demo

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
demo = build_demo(PROJECT)
if __name__=='__main__':
    demo.launch(server_name='0.0.0.0',server_port=int(os.environ.get('PORT','7860')),show_error=True)
