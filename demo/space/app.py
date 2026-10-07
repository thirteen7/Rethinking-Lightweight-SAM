"""Hugging Face Gradio launcher for the project's custom card interface."""
from pathlib import Path
import importlib.util
import subprocess
import sys

PROJECT = Path('/tmp/rethinking-lightweight-sam')
if not (PROJECT / 'demo/server.py').is_file():
    subprocess.run(['git', 'clone', '--depth', '1',
        'https://github.com/thirteen7/Rethinking-Lightweight-SAM.git', str(PROJECT)], check=True)
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location('project_demo_server', PROJECT / 'demo/server.py')
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)

import gradio as gr
with gr.Blocks(title='Rethinking Lightweight SAM', css='''
  .gradio-container {max-width: 1200px !important; padding: 8px !important;}
  footer {display: none !important;}
  .project-frame {width: 100%; height: 900px; border: none; border-radius: 14px;}
''') as demo:
    gr.HTML('<iframe class="project-frame" src="/interactive/?embed=1" '
            'title="Rethinking Lightweight SAM interactive application"></iframe>')

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, prevent_thread_lock=True)
    demo.server_app.mount('/interactive', server.create_app(device='cpu'))
    demo.block_thread()
