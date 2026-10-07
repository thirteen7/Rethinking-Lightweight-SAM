"""ZeroGPU interactive segmentation with the project's complete checkpoints."""
from pathlib import Path
import copy
import subprocess
import sys
import time

import spaces
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT = Path('/tmp/rethinking-lightweight-sam')
if not (PROJECT / 'demo/server.py').is_file():
    subprocess.run(['git', 'clone', '--depth', '1',
        'https://github.com/thirteen7/Rethinking-Lightweight-SAM.git', str(PROJECT)], check=True)
sys.path.insert(0, str(PROJECT))
from prompt_adaptive_sam import Predictor
from download_models import download, sha256
import json

torch.set_num_threads(2)
index = json.loads((PROJECT / 'models.json').read_text(encoding='utf-8'))
templates = {}
for name in ('tinysam', 'mobilesam'):
    entry = index['models'][name]
    target = PROJECT / 'weights' / entry['filename']
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{index['repository']}/releases/download/{index['release']}/{entry['filename']}"
    download(url, target, entry['sha256'], entry['bytes'])
    templates[name] = Predictor(target, device='cuda')


def new_state(image=None):
    return dict(image=image, points=[], box=None, corners=[], previous=None,
                embedding=None, image_tensor=None, model=None, rounds=0,
                previous_count=0, native_hw=None, input_hw=None, mask=None)


def render(state, show_mask=True):
    if state['image'] is None:
        return None
    image = Image.fromarray(state['image']).convert('RGBA')
    if show_mask and state.get('mask') is not None:
        mask = state['mask']
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., :3] = [118, 91, 234]
        rgba[..., 3] = mask.astype(np.uint8) * 112
        image = Image.alpha_composite(image, Image.fromarray(rgba, 'RGBA'))
    draw = ImageDraw.Draw(image)
    radius = max(4, int(max(image.size) / 95))
    if state.get('box'):
        draw.rectangle(tuple(state['box']), outline='#b9a4ff', width=max(2, radius // 3))
    for point in state['points']:
        x, y, label = point['x'], point['y'], point['label']
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill='#7860de' if label else '#d87865', outline='white', width=2)
        draw.line((x-radius*.45, y, x+radius*.45, y), fill='white', width=2)
        if label:
            draw.line((x, y-radius*.45, x, y+radius*.45), fill='white', width=2)
    for x, y in state['corners']:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='#b9a4ff', outline='white', width=2)
    return np.asarray(image.convert('RGB'))


def upload(image):
    if image is None:
        return new_state(), None, None, 'Upload an image to begin.'
    image = np.asarray(Image.fromarray(image).convert('RGB'))
    height, width = image.shape[:2]
    if max(height, width) > 1800:
        ratio = 1800 / max(height, width)
        image = np.asarray(Image.fromarray(image).resize((round(width*ratio), round(height*ratio))))
    state = new_state(image)
    return state, render(state), None, 'Image ready. Add a foreground point or two box corners.'


def clear():
    return new_state(), None, 'Upload an image to begin.'


def reset(state):
    state = new_state(state.get('image'))
    return state, render(state), None, 'Prompts reset. Select another object.'


def select_point(state, mode, label, event: gr.SelectData):
    if state['image'] is None:
        raise gr.Error('Upload an image first.')
    if state['rounds'] >= 3:
        raise gr.Error('Three rounds are complete. Reset prompts for another object.')
    x, y = map(float, event.index)
    if mode == 'Box' and state['rounds'] == 0:
        if state['box'] is not None:
            raise gr.Error('The box is ready. Predict before adding a corrective click.')
        state['corners'].append([x, y])
        if len(state['corners']) == 2:
            first, second = state['corners']
            box = [min(first[0], second[0]), min(first[1], second[1]),
                   max(first[0], second[0]), max(first[1], second[1])]
            if box[2]-box[0] < 3 or box[3]-box[1] < 3:
                state['corners'] = []
                raise gr.Error('Choose two distinct corners around the target.')
            state['box'] = box
            state['corners'] = []
            message = 'Box defined. Run prediction.'
        else:
            message = 'Select the opposite box corner.'
    else:
        if len(state['points']) > state['previous_count']:
            raise gr.Error('Predict this prompt before adding another click.')
        positive = label == 'Foreground (+)'
        if mode == 'Point' and state['rounds'] == 0 and not positive:
            raise gr.Error('Begin with a foreground click.')
        state['points'].append(dict(x=x, y=y, label=int(positive)))
        message = 'Prompt added. Run prediction.'
    return state, render(state, False), message


@spaces.GPU(duration=30)
def predict(state, model, mode):
    if state['image'] is None:
        raise gr.Error('Upload an image first.')
    if state['rounds'] >= 3:
        raise gr.Error('Three rounds are complete. Reset prompts for another object.')
    if mode == 'Box' and state['box'] is None:
        raise gr.Error('Select two corners to define a box.')
    expected = state['rounds'] + (1 if mode == 'Point' else 0)
    if len(state['points']) != expected:
        raise gr.Error('Add one corrective click for the next round.')
    started = time.perf_counter()
    name = model.lower()
    predictor = copy.copy(templates[name])
    if state['embedding'] is None or state['model'] != name:
        predictor.set_image(state['image'])
        state['native_hw'], state['input_hw'] = predictor.native_hw, predictor.input_hw
    else:
        predictor.embedding = state['embedding'].to('cuda')
        predictor.image = state['image_tensor'].to('cuda')
        predictor.native_hw, predictor.input_hw = state['native_hw'], state['input_hw']
    coordinates = [[row['x'], row['y']] for row in state['points']]
    labels = [row['label'] for row in state['points']]
    mask, logits, choice = predictor.predict(coordinates, labels,
        box=state['box'] if mode == 'Box' else None, previous=state['previous'])
    if not np.isfinite(logits).all():
        raise gr.Error('The prediction failed the finite-output check.')
    state['embedding'] = predictor.embedding.detach().cpu()
    state['image_tensor'] = predictor.image.detach().cpu()
    state['previous'] = logits
    state['model'] = name
    state['mask'] = mask
    state['rounds'] += 1
    state['previous_count'] = len(state['points'])
    predictor.model.last_trace = None
    elapsed = time.perf_counter() - started
    message = f"**Round {state['rounds']} / 3** · Candidate {choice} · {elapsed:.2f} s on ZeroGPU"
    if state['rounds'] < 3:
        message += '\n\nAdd a foreground/background correction, then predict again.'
    else:
        message += '\n\nComplete. Reset prompts to select another object.'
    return state, render(state, False), render(state), message


CSS = '''
body {background:#f6f7fb!important;}
.gradio-container {max-width:1120px!important;margin:auto!important;font-family:Inter,"Segoe UI",Arial,sans-serif!important;}
.paper-header {padding:28px 10px 24px;}
.paper-eyebrow {font-size:10px;letter-spacing:.15em;color:#8d839f;margin-bottom:14px;}
.paper-header h1 {font-size:38px;font-weight:650;letter-spacing:-.045em;line-height:1.15;color:#222638;}
.paper-header h1 span {color:#7860de;}
.paper-header p {font-size:14px;color:#8a90a0;margin-top:12px;line-height:1.8;}
.paper-links {display:flex;gap:10px;margin-top:20px;}
.paper-links a {text-decoration:none!important;padding:9px 15px;border:1px solid #e5e7ef;border-radius:9px;color:#596078!important;background:#fff;font-size:11px;}
.paper-links a:first-child {background:#24283b;color:white!important;border-color:#24283b;}
#controls {background:#fff;border:1px solid #e5e7ef;border-radius:18px;padding:23px;gap:16px;}
#canvas-panel {background:#fff;border:1px solid #e5e7ef;border-radius:18px;padding:19px;}
.block {border-radius:12px!important;border-color:#e5e7ef!important;box-shadow:none!important;}
#predict {background:#7860de!important;color:white!important;border:0!important;border-radius:9px!important;}
#status {background:#f7f5fd;padding:16px 20px;border:1px solid #e8e1f6;border-radius:13px;font-size:12px;}
.demo-note {font-size:11px;color:#9da3b1;line-height:1.8;}
footer {display:none!important;}
'''
HEADER = '''<div class="paper-header"><div class="paper-eyebrow">RETHINKING LIGHTWEIGHT SAM / INTERACTIVE APPLICATION</div>
<h1>Prompt-Adaptive <span>Refinement.</span></h1><p>Upload an image. Define the target. Refine with corrective interaction.</p>
<div class="paper-links"><a href="https://thirteen7.github.io/Rethinking-Lightweight-SAM/" target="_blank">Project page ↗</a><a href="https://github.com/thirteen7/Rethinking-Lightweight-SAM" target="_blank">Code &amp; models ↗</a></div></div>'''

with gr.Blocks(title='Rethinking Lightweight SAM', css=CSS, theme=gr.themes.Soft(primary_hue='violet', neutral_hue='slate'), delete_cache=(900, 900)) as demo:
    gr.HTML(HEADER)
    session = gr.State(new_state(), time_to_live=1200)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=235, elem_id='controls'):
            gr.Markdown('### Configuration')
            model = gr.Dropdown(['TinySAM', 'MobileSAM'], value='TinySAM', label='Backbone')
            mode = gr.Radio(['Point', 'Box'], value='Point', label='Prompt mode')
            label = gr.Radio(['Foreground (+)', 'Background (−)'], value='Foreground (+)', label='Correction label')
            gr.HTML('<div class="demo-note">Point: select one foreground click.<br>Box: select two opposite corners.<br>After prediction, add one correction per round.</div>')
            run = gr.Button('Predict mask →', variant='primary', elem_id='predict')
            reset_button = gr.Button('Reset prompts')
            gr.HTML('<div class="demo-note">Three interaction rounds · ZeroGPU inference<br>Uploads and predictions use temporary Space storage.<br>Uploaded images have no ground-truth IoU score.</div>')
        with gr.Column(scale=3, min_width=280, elem_id='canvas-panel'):
            with gr.Row():
                prompt_image = gr.Image(label='Image & prompts', type='numpy', sources=['upload'], height=390)
                output = gr.Image(label='Refined prediction', type='numpy', interactive=False, height=390)
            status = gr.Markdown('Upload an image to begin.', elem_id='status')
    prompt_image.upload(upload, inputs=[prompt_image], outputs=[session, prompt_image, output, status])
    prompt_image.select(select_point, inputs=[session, mode, label], outputs=[session, prompt_image, status])
    prompt_image.clear(clear, outputs=[session, output, status])
    reset_button.click(reset, inputs=[session], outputs=[session, prompt_image, output, status])
    mode.change(reset, inputs=[session], outputs=[session, prompt_image, output, status])
    model.change(reset, inputs=[session], outputs=[session, prompt_image, output, status])
    run.click(predict, inputs=[session, model, mode], outputs=[session, prompt_image, output, status], api_name='predict_mask', concurrency_limit=1)
demo.queue(default_concurrency_limit=1, max_size=12)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
