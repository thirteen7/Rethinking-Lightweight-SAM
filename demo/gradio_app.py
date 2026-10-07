"""Shared SegAny / SegEvery research workspace."""
from pathlib import Path
import base64
import copy
import html
import io
import json
import logging
import os
import time
import subprocess
import sys
from urllib.parse import urlencode, urlsplit

DEVICE = os.environ.get('SAM_DEMO_DEVICE', 'cuda')
ANIMATION_URL = os.environ.get('SAM_DEMO_ANIMATION_URL',
    'https://thirteen7.github.io/Rethinking-Lightweight-SAM/assets/fsd-animation.html')
if DEVICE == 'cuda' and os.environ.get('SAM_DEMO_LOCAL') != '1':
    import spaces  # Initialize ZeroGPU before torch.
    gpu = spaces.GPU
else:
    def gpu(**unused):
        return lambda fn: fn

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
from prompt_adaptive_sam import Predictor
from prompt_adaptive_sam.everything import render_instances
from download_models import download
from demo.comparison import compare_everything, compare_prompt_stage

LOG = logging.getLogger('segmentation-workspace')
TEMPLATES = {}


def empty_state(image=None):
    return dict(image=image, points=[], box=None, corners=[], mask=None,
                instances=[], details=None, comparison=None,
                comparison_runs=0, prompt_comparison=None, prompt_stage=0)


def fsd_walkthrough(details=None, *, image=None, dense_details=None, overlays=None):
    """Show actual prompt coordinates and retained requests from this run."""
    if details and details.get('method')=='dense':
        return '<div class="dense-walkthrough"><span>DENSE SAM / NATIVE PATH</span><h2>Complete every request.</h2><p>Image encoding → independent point prompts → full native decoding → SAM filters &amp; NMS</p><b>'+str(int(details['native_points']))+' / '+str(int(details['total_points']))+' completed requests</b><p>Dense SAM completes every sampled prompt. It does not use FSD selection.</p></div>'
    url = ANIMATION_URL
    attributes = ''
    if details is not None:
        url += '?' + urlencode(dict(live=1))
        selection = details.get('selection')
        if selection is not None and image is not None:
            def thumbnail(array):
                photo = Image.fromarray(np.asarray(array,dtype=np.uint8)).convert('RGB')
                photo.thumbnail((512,512))
                stream = io.BytesIO();photo.save(stream,format='JPEG',quality=82)
                return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')
            trace = dict(source='live',grid=details['grid'],image_hw=details['image_hw'],
                points=details['point_grid'],selected=details['native_point_indices'],
                waves=selection['wave_source_points'],guards=selection['guard_source_points'],
                guard_only=selection['guard_only_source_points'],image=thumbnail(image),
                masks=dict(dense=dense_details['masks'] if dense_details else None,fsd=details['masks']))
            if overlays is not None:
                trace['overlays'] = dict(zip(('dense','fsd'),map(thumbnail,overlays)))
            attributes = ' data-fsd-trace="'+html.escape(json.dumps(trace,separators=(',',':')),quote=True)+'"'
    return '<iframe class="fsd-embedded-map" title="Interactive FSD-SAM decoding walkthrough" src="'+html.escape(url,quote=True)+'"'+attributes+' loading="lazy"></iframe>'


def grid_view(state, grid=16):
    if state['image'] is None:
        return None
    image = Image.fromarray(state['image']).convert('RGB')
    draw = ImageDraw.Draw(image)
    w,h = image.size
    radius = max(2, max(w,h)/220)
    for y in range(int(grid)):
        for x in range(int(grid)):
            px,py = (x+.5)*w/int(grid),(y+.5)*h/int(grid)
            draw.ellipse((px-radius,py-radius,px+radius,py+radius),fill='white',outline='#319995',width=1)
    return np.asarray(image)


def prompt_view(state):
    if state['image'] is None:
        return None
    image = Image.fromarray(state['image']).convert('RGB')
    draw = ImageDraw.Draw(image)
    radius = max(4, round(max(image.size)/100))
    if state.get('box'):
        draw.rectangle(tuple(state['box']), outline='#a895f1', width=max(2, radius//2))
    for point in state['points']:
        x, y = point['x'], point['y']
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),
                     fill='#7960df' if point['label'] else '#e07a68', outline='white', width=2)
        draw.line((x-radius*.4,y,x+radius*.4,y), fill='white', width=2)
        if point['label']:
            draw.line((x,y-radius*.4,x,y+radius*.4), fill='white', width=2)
    for x,y in state['corners']:
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill='#a895f1',outline='white',width=2)
    return np.asarray(image)


def output_view(state, opacity=.48, selection='All instances'):
    if state['image'] is None:
        return None
    if state['details'] is not None:
        chosen = None if selection=='All instances' else int(selection.split()[-1])-1
        return render_instances(state['image'],state['instances'],float(opacity),chosen)
    if state['mask'] is None:
        return None
    image = prompt_view(state).astype(np.float32)
    mask = state['mask']
    image[mask] = image[mask]*(1-opacity)+np.asarray([118,91,234])*opacity
    return image.astype(np.uint8)


def metrics(mode='READY', masks='—', requests='—', elapsed='—'):
    fields = [('MODE',mode),('INSTANCES',masks),('PROMPTS / STAGE',requests),('INFERENCE',elapsed)]
    return '<div class="metrics-row">'+''.join(
        f'<div class="metric-tile"><span>{label}</span><strong>{html.escape(str(value))}</strong></div>'
        for label,value in fields)+'</div>'


def comparison_duration(state, grid):
    # Reserve more time for dense grids and large uploads; keep small demos
    # within the visitor's available ZeroGPU quota.
    image = state.get('image')
    large_image = image is not None and image.shape[0]*image.shape[1]>1_500_000
    return 60 if large_image else 30


@gpu(duration=comparison_duration)
@torch.inference_mode()
def compute_comparison(state, grid):
    return compare_everything(TEMPLATES['vith'], state['image'], int(grid),
                              run_index=state.get('comparison_runs', 0))


@gpu(duration=15)
@torch.inference_mode()
def compute_prompt_comparison(state, name, prompt, previous_pair=None):
    return compare_prompt_stage(TEMPLATES[name], name, state['image'],
                                prompt, state['truth'], previous_pair)


def pair_metrics(pair=None):
    if pair is None:
        return '<div class="pair-times"><div><span>SAM ViT-H / DENSE</span><strong>— <small>ms</small></strong><p>Run both models to measure</p></div><div><span>FSD-SAM / ViT-H</span><strong>— <small>ms</small></strong><p>Run both models to measure</p></div></div>'
    cards = []
    for method,title in [('dense','SAM ViT-H / DENSE'),('fsd','FSD-SAM / ViT-H')]:
        row = pair['results'][method]
        cards.append(f'<div><span>{title}</span><strong>{row["total_ms"]:,.1f} <small>ms</small></strong><p>{row["info"]["masks"]} masks · {row["info"]["native_points"]} / {row["info"]["total_points"]} completed requests</p><p>Encode {row["encoding_ms"]:,.1f} ms + masks {row["generation_ms"]:,.1f} ms</p></div>')
    delta = pair['results']['dense']['total_ms']-pair['results']['fsd']['total_ms']
    difference = f'FSD-SAM uses {abs(delta):,.1f} ms '+('less' if delta>=0 else 'more')+' on this run'
    return '<div class="pair-times">'+''.join(cards)+'</div><div class="pair-scope">'+difference+' · '+html.escape(pair['device'])+' · FP32<br>Same frozen ViT-H, image, grid and filters. Encoding + mask generation; loading, warm-up, queue and overlay rendering excluded.</div>'


def comparison_views(state, opacity=.48):
    pair = state.get('comparison')
    if pair is None:
        return None,None
    return tuple(render_instances(state['image'],pair['results'][key]['masks'],float(opacity)) for key in ('dense','fsd'))


def prompt_overlay(state, row, field, opacity):
    picture = np.asarray(state['image'],dtype=np.float32).copy()
    mask = row[field]['mask']
    color = np.array([49,151,151] if field=='original' else [118,91,234])
    picture[mask] = picture[mask]*(1-float(opacity))+color*float(opacity)
    view_state = dict(state,image=np.clip(picture,0,255).astype(np.uint8),points=row['points'],box=row['box'],corners=[])
    return prompt_view(view_state)


def prompt_views(state, opacity=.48):
    pair = state.get('prompt_comparison')
    if pair is None:
        return None,None
    index = max(0,min(int(state.get('prompt_stage',len(pair['stages'])-1)),len(pair['stages'])-1))
    return tuple(prompt_overlay(state,pair['stages'][index],field,opacity) for field in ('original','refined'))


def prompt_metrics(pair=None,selected_stage=None):
    if pair is None:
        return '<div class="prompt-result-intro">Run the initial prompt, then add one foreground point at a time.</div>'
    rows = []
    selected_stage = len(pair['stages'])-1 if selected_stage is None else selected_stage
    for stage,row in enumerate(pair['stages']):
        original,refined = row['original']['iou_percent'],row['refined']['iou_percent']
        label = 'Initial prompt' if stage==0 else f'+{stage} point'+('s' if stage>1 else '')
        current = ' class="current-stage"' if stage==selected_stage else ''
        rows.append(f'<tr{current}><td>{label}</td><td>{original:.2f}%</td><td><b>{refined:.2f}%</b></td><td>{refined-original:+.2f} pp</td></tr>')
    return '<div class="prompt-score-table"><table><thead><tr><th>Completed stage</th><th>Original IoU</th><th>Refined IoU</th><th>Difference</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table><p>Measured target IoU · same prompts · each model keeps its own mask feedback.</p></div>'


def prompt_details(state):
    pair = state.get('prompt_comparison')
    if pair is None:
        return None
    result = {key:value for key,value in pair.items() if key not in ('_feedback','stages')}
    result['stages'] = [dict(row,original={k:v for k,v in row['original'].items() if k!='mask'},
        refined={k:v for k,v in row['refined'].items() if k!='mask'}) for row in pair['stages']]
    result['example'] = state['example_id']
    result['shown_stage'] = int(state.get('prompt_stage',len(pair['stages'])-1))
    return result


def run_workspace(state, model, mode, grid, opacity):
    if state['image'] is None:
        raise gr.Error('Choose an image first.')
    try:
        if mode!='Everything':
            if not state.get('curated_trajectory') or state.get('curated_model')!=model.lower() or state.get('curated_mode')!=mode.lower():
                raise gr.Error('Select a curated point/box example first.')
            # The first run contains only the initial prompt. Corrections are
            # separate callbacks, so future prompts cannot leak into this view.
            state = dict(state)
            prompt = state['curated_trajectory'][0]
            pair = compute_prompt_comparison(state,model.lower(),prompt)
            state['prompt_comparison'] = pair
            state['prompt_stage'] = 0
            state['points'],state['box'] = copy.deepcopy(prompt['points']),copy.deepcopy(prompt['box'])
            LOG.info('prompt comparison example=%s model=%s mode=%s stage=0 iou=%s',state['example_id'],model,mode,
                [(r['original']['iou_percent'],r['refined']['iou_percent']) for r in pair['stages']])
            return state
        pair = compute_comparison(state,grid)
        state = dict(state)
        state['comparison'],state['comparison_runs'] = pair,state.get('comparison_runs',0)+1
        right = pair['results']['fsd']
        state['instances'],state['details'] = right['masks'],right['info']
        dense_ms,fsd_ms = [pair['results'][key]['total_ms'] for key in ('dense','fsd')]
        LOG.info('paired result device=%s grid=%d retained=%d/%d dense_ms=%.3f fsd_ms=%.3f',pair['device'],int(grid),state['details']['native_points'],state['details']['total_points'],dense_ms,fsd_ms)
        return state
    except Exception as error:
        LOG.exception('comparison failed')
        if isinstance(error,gr.Error):
            raise
        raise gr.Error(str(error)[:300]) from error


def advance_workspace(state, model, mode, correction):
    if mode=='Everything' or state.get('curated_model')!=model.lower() or state.get('curated_mode')!=mode.lower():
        raise gr.Error('Choose a curated Point or Box example first.')
    previous = state.get('prompt_comparison')
    if previous is None:
        raise gr.Error('Run the initial comparison before adding a correction.')
    if correction not in (1,2) or len(previous['stages'])!=correction:
        raise gr.Error('Add corrections in order: first +1 point, then +2 points.')
    prompt = state['curated_trajectory'][correction]
    try:
        pair = compute_prompt_comparison(state,model.lower(),prompt,previous)
    except Exception as error:
        LOG.exception('corrective comparison failed')
        raise gr.Error(str(error)[:300]) from error
    result = dict(state,prompt_comparison=pair,prompt_stage=correction,points=copy.deepcopy(prompt['points']),box=copy.deepcopy(prompt['box']))
    current = pair['stages'][-1]
    LOG.info('prompt correction example=%s model=%s stage=%d original_iou=%.3f refined_iou=%.3f',
        state['example_id'],model,correction,current['original']['iou_percent'],current['refined']['iou_percent'])
    return result


def browse_workspace(state, offset):
    pair = state.get('prompt_comparison')
    if pair is None:
        raise gr.Error('Run the initial comparison before browsing saved results.')
    index = int(state.get('prompt_stage',len(pair['stages'])-1))+offset
    if not 0<=index<len(pair['stages']):
        raise gr.Error('Only completed prompt stages can be viewed.')
    row = pair['stages'][index]
    return dict(state,prompt_stage=index,points=copy.deepcopy(row['points']),box=copy.deepcopy(row['box']))


def build_demo(project):
    torch.set_num_threads(4 if DEVICE=='cpu' else 2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    index = json.loads((project/'models.json').read_text())
    for name in ('tinysam','mobilesam'):
        entry = index['models'][name]
        path = project/'weights'/entry['filename']
        path.parent.mkdir(exist_ok=True)
        download(f"https://github.com/{index['repository']}/releases/download/{index['release']}/{entry['filename']}",path,entry['sha256'],entry['bytes'])
        TEMPLATES[name] = Predictor(path,device=DEVICE)
    subprocess.run([sys.executable,str(project/'download_models.py'),'--model','vith','--output',str(project/'weights')],check=True)
    TEMPLATES['vith'] = Predictor(project/'weights'/index['models']['vith']['filename'],device=DEVICE)
    css = (project/'demo/space/workspace.css').read_text()
    records = json.loads((project/'site/examples.json').read_text())
    gallery_rows = [row for row in records['examples'] if row.get('foreground_demo')]
    default_image = np.asarray(Image.open(project/'site/assets/examples/fruit.jpg').convert('RGB'))
    default_state = empty_state(default_image)
    title = 'Rethinking Lightweight SAM with Prompt-Adaptive Refinement and Efficient Segment Everything Inference'
    header = '<div class="studio-header"><div><span class="studio-eyebrow">RESEARCH STUDIO</span><h1>'+title+'</h1><p>Compare the same prompt on both models, then add corrections one point at a time.</p><div class="studio-links"><a href="https://github.com/thirteen7/Rethinking-Lightweight-SAM" target="_blank" rel="noopener">Code ↗</a><a href="https://thirteen7.github.io/Rethinking-Lightweight-SAM/" target="_blank" rel="noopener">Project page ↗</a></div></div><div class="studio-summary"><span class="studio-summary-label">ONE IMAGE / TWO PATHS</span><div><b>Point &amp; Box</b><span>Initial → +1 point → +2 points</span></div><div><b>Everything</b><span>ViT-H / FSD-SAM · measured ms</span></div></div></div>'
    source = urlsplit(ANIMATION_URL)
    origin = json.dumps(source.scheme+'://'+source.netloc)
    js = """() => {
        const origin=ORIGIN;
        window.addEventListener('message',event=>{
            if(event.origin!==origin || event.data?.type!=='fsd-animation-size')return;
            const height=Number(event.data.height);
            if(!Number.isFinite(height)||height<250||height>4000)return;
            document.querySelectorAll('iframe.fsd-embedded-map').forEach(frame=>{
                if(event.source===frame.contentWindow)frame.style.setProperty('height',height+'px','important');
            });
        });
        const seen=new WeakSet();
        const sync=()=>document.querySelectorAll('iframe.fsd-embedded-map[data-fsd-trace]').forEach(frame=>{
            if(seen.has(frame))return;seen.add(frame);
            const send=()=>{try{frame.contentWindow.postMessage({type:'fsd-live-trace',trace:JSON.parse(frame.dataset.fsdTrace)},origin);}catch(error){console.warn('FSD point-map transfer failed',error);}};
            frame.addEventListener('load',send);send();
        });
        new MutationObserver(sync).observe(document.body,{childList:true,subtree:true});sync();
    }""".replace('ORIGIN',origin)
    with gr.Blocks(title=title,css=css,js=js,theme=gr.themes.Base(primary_hue='violet',neutral_hue='slate'),delete_cache=(900,900)) as demo:
        gr.HTML(header)
        with gr.Tabs(elem_id='studio-tabs'):
            with gr.Tab('Live studio'):
                session = gr.State(default_state,time_to_live=1200)
                with gr.Row(elem_id='task-bar'):
                    mode = gr.Radio(['Point','Box','Everything'],value='Everything',label='Segmentation mode',elem_id='mode-switch')
                    model = gr.Dropdown(['TinySAM','MobileSAM'],value='TinySAM',label='Lightweight backbone',visible=False,elem_id='model-select')
                stats = gr.HTML(metrics('ViT-H vs FSD','—','32 × 32','Ready'),elem_id='metric-cards')
                with gr.Row(equal_height=False,elem_id='workspace-row'):
                    with gr.Column(scale=1,min_width=220,elem_id='settings-card'):
                        gr.Markdown('### Image & prompts')
                        image = gr.Image(value=grid_view(default_state,32),type='numpy',label='Input image · uploads welcome',sources=['upload'],height=190,elem_id='input-image')
                        with gr.Group(elem_id='every-controls') as every_controls:
                            grid = gr.Radio([8,16,32],value=32,label='Shared points per side')
                            gr.Markdown('Same image, grid and frozen ViT-H. One run measures both methods.',elem_id='every-note')
                        with gr.Group(visible=False,elem_id='point-controls') as point_controls:
                            gr.Markdown('Run the initial prompt, then add the two preset foreground corrections one at a time. Browse saved results with **Previous / Next**.',elem_id='point-note')
                        opacity = gr.Slider(0,1,value=.48,step=.01,label='Overlay opacity')
                        button = gr.Button('Run both models →',variant='primary',elem_id='run-segmentation')
                        with gr.Row(visible=False,elem_id='correction-actions') as correction_actions:
                            add_first = gr.Button('+1 point',interactive=False,size='sm',scale=1,min_width=85,elem_id='add-first-point')
                            add_second = gr.Button('+2 points',interactive=False,size='sm',scale=1,min_width=85,elem_id='add-second-point')
                        reset_button = gr.Button('Reset comparison',elem_id='reset-target')
                        gr.HTML('<p class="studio-note">Curated Point / Box targets · foreground clicks.<br>Everything supports uploads · FP32.</p>')
                    with gr.Column(scale=3,min_width=300,elem_id='canvas-card'):
                        with gr.Group(visible=False,elem_id='prompt-pair-results') as prompt_group:
                            with gr.Row(elem_id='prompt-stage-bar'):
                                stage_title = gr.HTML('<div class="prompt-stage-heading"><b>Initial prompt</b><span>Original / refined · same prompt</span></div>',elem_id='current-stage-title')
                                with gr.Row(elem_id='history-actions'):
                                    previous_result = gr.Button('← Previous result',interactive=False,size='sm',min_width=110,elem_id='previous-result')
                                    next_result = gr.Button('Next result →',interactive=False,size='sm',min_width=110,elem_id='next-result')
                            with gr.Row(equal_height=True,elem_id='prompt-image-row'):
                                original = gr.Image(label='Original · initial prompt',interactive=False,type='numpy',height=300,min_width=120,elem_id='prompt-original')
                                refined = gr.Image(label='Refined · initial prompt',interactive=False,type='numpy',height=300,min_width=120,elem_id='prompt-refined')
                            point_scores = gr.HTML(prompt_metrics(),elem_id='prompt-iou-results')
                        with gr.Group(elem_id='every-pair-results') as pair_group:
                            with gr.Row(equal_height=True,elem_id='every-image-row'):
                                pair_left = gr.Image(value=default_image,type='numpy',label='SAM ViT-H / Dense · ready',interactive=False,height=340,min_width=120,elem_id='every-dense')
                                pair_right = gr.Image(value=default_image,type='numpy',label='FSD-SAM / ViT-H · ready',interactive=False,height=340,min_width=120,elem_id='every-fsd')
                            times = gr.HTML(pair_metrics(),elem_id='pair-timing')
                        pair_details = gr.JSON(visible=False,elem_id='comparison-details')
                        status = gr.Markdown('**Orange bowl is ready.** Press **Run both models** for both masks and measured times.',elem_id='studio-status')
                        with gr.Accordion('FSD-SAM instance details',open=False,elem_id='instance-inspector') as inspector:
                            selected = gr.Dropdown(['All instances'],value='All instances',label='Inspect FSD instance')
                            table = gr.Dataframe(headers=['Instance','Area (px)','SAM predicted IoU','Stability'],datatype=['str','number','number','number'],interactive=False,wrap=True)
                with gr.Group(visible=False,elem_id='curated-gallery') as curated_gallery:
                    gallery = gr.Gallery([(str(project/'site'/row['image']),row['title']+' · '+row['demo_model']+' / '+row['demo_prompt']) for row in gallery_rows],label='Curated comparisons · select a thumbnail',columns=4,height=150,object_fit='cover',allow_preview=False)
                with gr.Group(elem_id='everything-gallery') as every_gallery:
                    every_example_input = gr.Image(visible=False,type='numpy')
                    every_examples = gr.Examples(examples=[str(project/'site/assets/examples/fruit.jpg'),str(project/'site/assets/examples/bear.jpg')],inputs=[every_example_input],label='Everything examples')
            with gr.Tab('How FSD works'):
                gr.Markdown('Follow the same two-dimensional grid through encoding, preview selection and mask completion. After an Everything run, this diagram uses the actual points selected on your image.',elem_id='flow-introduction')
                diagram = gr.HTML(fsd_walkthrough(),elem_id='fsd-workspace')
        gr.HTML('<div class="studio-footer"><b>'+title+'</b><a href="https://github.com/thirteen7/Rethinking-Lightweight-SAM/blob/main/docs/results.md" target="_blank" rel="noopener">Paper results ↗</a></div>')
        outputs = [session,image,original,refined,status,stats,table,selected,diagram,pair_left,pair_right,times,pair_details,point_scores,
            stage_title,add_first,add_second,button,previous_result,next_result,correction_actions,every_controls,point_controls,inspector,prompt_group,pair_group,model,curated_gallery,every_gallery]

        def render_workspace(state,value,grid_size,alpha=.48):
            enabled = value=='Everything'
            pair = state.get('prompt_comparison') if not enabled else None
            every_pair = state.get('comparison') if enabled else None
            name = {'tinysam':'TinySAM','mobilesam':'MobileSAM'}.get(state.get('curated_model'),'Model')
            count = len(pair['stages']) if pair else 0
            index = max(0,min(int(state.get('prompt_stage',count-1)),count-1)) if pair else 0
            stage = 'Initial prompt' if index==0 else f'+{index} point'+('s' if index>1 else '')
            view = grid_view(state,grid_size) if enabled else prompt_view(state)
            left,right = comparison_views(state,alpha) if every_pair else (state['image'],state['image'])
            prompt_left,prompt_right = prompt_views(state,alpha)
            prompt_left = view if prompt_left is None and not enabled else prompt_left
            prompt_right = view if prompt_right is None and not enabled else prompt_right
            scores = [[f'Instance {i+1:02d}',m['area'],round(m['predicted_iou'],3),round(m['stability_score'],3)] for i,m in enumerate(state['instances'])] if every_pair else []
            if enabled:
                message = '**Image ready.** Run both models to measure the two generation paths.'
                if every_pair:
                    dense_ms,fsd_ms = [every_pair['results'][key]['total_ms'] for key in ('dense','fsd')]
                    message = f'**Both models finished.** SAM ViT-H: **{dense_ms:,.1f} ms** · FSD-SAM: **{fsd_ms:,.1f} ms**. Same {int(grid_size)} × {int(grid_size)} grid.'
            elif pair:
                next_step = 'Add **+1 point** to continue.' if count==1 else 'Add **+2 points** to continue.' if count==2 else 'Two corrections complete. Reset to start again.'
                if index<count-1:
                    next_step = 'Viewing a saved result. The next correction continues from the latest completed stage.' if count<3 else 'Viewing a saved result. Use Next to compare a later stage.'
                message = f'**{state["example_title"]} · {stage}.** Both models received the same foreground prompts. {next_step}'
            else:
                message = f'**{state["example_title"]} is ready.** Run the initial prompt first; corrections will appear one at a time.'
            label = lambda field: field.title()+(f' · {pair["stages"][index][field]["iou_percent"]:.2f}% IoU' if pair else ' · ready')
            result = {
                session:state,image:gr.update(value=view,interactive=enabled,height=190,label='Input image · uploads welcome' if enabled else 'Curated target · current prompts'),
                original:gr.update(value=prompt_left,label=label('original')),refined:gr.update(value=prompt_right,label=label('refined')),
                status:message,stats:metrics('ViT-H vs FSD' if enabled else name,len(scores) if every_pair else 'One target' if pair else '—',
                    f'{int(grid_size)} × {int(grid_size)}' if enabled else stage,'Measured / ms' if every_pair else 'True IoU' if pair else 'Ready'),
                table:scores,selected:gr.update(choices=['All instances']+[r[0] for r in scores],value='All instances'),
                diagram:fsd_walkthrough(state['details'],image=state['image'],dense_details=every_pair['results']['dense']['info'],overlays=(left,right)) if every_pair else gr.skip(),
                pair_left:gr.update(value=left if enabled else None,label='SAM ViT-H / Dense'+(' · measured' if every_pair else ' · ready')),
                pair_right:gr.update(value=right if enabled else None,label='FSD-SAM / ViT-H'+(' · measured' if every_pair else ' · ready')),
                times:pair_metrics(every_pair),pair_details:every_pair if enabled else prompt_details(state),point_scores:prompt_metrics(pair,index),
                stage_title:'<div class="prompt-stage-heading"><b>'+stage+'</b><span>'+(f'Saved result {index+1} / {count} · ' if pair else '')+'Original / refined · same prompt</span></div>',
                add_first:gr.update(interactive=count==1,variant='primary' if count==1 else 'secondary'),
                add_second:gr.update(interactive=count==2,variant='primary' if count==2 else 'secondary'),
                button:gr.update(interactive=enabled or count==0,value='Run both models →' if enabled or count==0 else 'Initial comparison complete'),
                previous_result:gr.update(interactive=bool(pair) and index>0),
                next_result:gr.update(interactive=bool(pair) and index<count-1),
                correction_actions:gr.update(visible=not enabled),
            }
            for component,show in zip([every_controls,point_controls,inspector,prompt_group,pair_group,model,curated_gallery,every_gallery],
                [enabled,not enabled,enabled,not enabled,enabled,not enabled,not enabled,enabled]):
                result[component] = gr.update(visible=show)
            return result

        def load_image(value,current_mode='Everything',grid_size=32):
            if current_mode!='Everything':
                raise gr.Error('Point and Box use the curated annotated samples. Uploads are available in Everything.')
            if value is None:
                return render_workspace(empty_state(),'Everything',grid_size)
            photo = Image.fromarray(np.asarray(value,dtype=np.uint8)).convert('RGB')
            photo.thumbnail((1800,1800))
            return render_workspace(empty_state(np.asarray(photo).copy()),'Everything',grid_size)

        def switch(state,value,backbone='TinySAM',grid_size=32):
            enabled = value=='Everything'
            if enabled:
                state = empty_state(state.get('image') if state.get('curated_trajectory') is None and state.get('image') is not None else default_image)
            else:
                key = records['default_examples'][backbone.lower()][value.lower()]
                row = next(r for r in records['examples'] if r['id']==key)
                state = empty_state(np.asarray(Image.open(project/'site'/row['image']).convert('RGB')))
                state.update(example_id=row['id'],example_title=row['title'],curated_model=backbone.lower(),curated_mode=value.lower(),
                    curated_trajectory=row['trajectories'][backbone.lower()][value.lower()],truth=np.asarray(Image.open(project/'site'/row['gt_mask']))>0)
                state['points'],state['box'] = state['curated_trajectory'][0]['points'],state['curated_trajectory'][0]['box']
            return render_workspace(state,value,grid_size)

        def reset_workspace(state,current_mode,backbone,grid_size):
            return switch(state,current_mode,backbone,grid_size)

        image.upload(load_image,inputs=[image,mode,grid],outputs=outputs,api_name='load_image')
        image.clear(lambda: load_image(None),outputs=outputs,api_name=False)
        mode.change(switch,inputs=[session,mode,model,grid],outputs=outputs,api_name='switch_mode')
        model.change(switch,inputs=[session,mode,model,grid],outputs=outputs,api_name=False)
        reset_button.click(reset_workspace,inputs=[session,mode,model,grid],outputs=outputs,api_name='reset_image')
        grid.change(reset_workspace,inputs=[session,mode,model,grid],outputs=outputs,api_name=False)
        def segment(state,backbone,value,size,alpha):
            return render_workspace(run_workspace(state,backbone,value,size,alpha),value,size,alpha)
        def add_correction(state,backbone,value,size,alpha,number):
            return render_workspace(advance_workspace(state,backbone,value,number),value,size,alpha)
        button.click(segment,inputs=[session,model,mode,grid,opacity],outputs=outputs,api_name='segment',concurrency_limit=1)
        add_first.click(lambda state,backbone,value,size,alpha:add_correction(state,backbone,value,size,alpha,1),
            inputs=[session,model,mode,grid,opacity],outputs=outputs,api_name='add_correction_1',concurrency_limit=1)
        add_second.click(lambda state,backbone,value,size,alpha:add_correction(state,backbone,value,size,alpha,2),
            inputs=[session,model,mode,grid,opacity],outputs=outputs,api_name='add_correction_2',concurrency_limit=1)
        def browse_result(state,value,size,alpha,offset):
            return render_workspace(browse_workspace(state,offset),value,size,alpha)
        previous_result.click(lambda state,value,size,alpha:browse_result(state,value,size,alpha,-1),
            inputs=[session,mode,grid,opacity],outputs=outputs,api_name='previous_result')
        next_result.click(lambda state,value,size,alpha:browse_result(state,value,size,alpha,1),
            inputs=[session,mode,grid,opacity],outputs=outputs,api_name='next_result')

        def choose_example(state,grid_size,event:gr.SelectData):
            row = gallery_rows[int(event.index)]
            new_mode,new_model = row['demo_prompt'].title(),row['demo_model']
            result = switch(state,new_mode,new_model,grid_size)
            result[mode],result[model] = gr.update(value=new_mode),gr.update(value=new_model,visible=True)
            return result
        gallery.select(choose_example,inputs=[session,grid],outputs=outputs+[mode],api_name=False)
        every_example_input.change(lambda value,size: load_image(value,'Everything',size),inputs=[every_example_input,grid],outputs=outputs,api_name=False)

        def opacity_changed(state,alpha,selection):
            left,right = comparison_views(state,alpha)
            a,b = prompt_views(state,alpha)
            if state.get('curated_trajectory') and a is None:
                a = b = prompt_view(state)
            if state.get('comparison') is None:
                left = right = state['image']
            return a,b,left,right
        opacity.change(opacity_changed,inputs=[session,opacity,selected],outputs=[original,refined,pair_left,pair_right],queue=False,api_name=False)
        selected.change(lambda state,alpha,selection: output_view(state,alpha,selection),inputs=[session,opacity,selected],outputs=[pair_right],queue=False,api_name=False)
    return demo.queue(default_concurrency_limit=1,max_size=12)
