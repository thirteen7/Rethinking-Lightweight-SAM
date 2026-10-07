"""Shared SegAny / SegEvery research workspace."""
from pathlib import Path
import copy
import html
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
from demo.comparison import compare_everything, compare_prompts

LOG = logging.getLogger('segmentation-workspace')
TEMPLATES = {}


def empty_state(image=None):
    return dict(image=image, points=[], box=None, corners=[], mask=None,
                instances=[], details=None, comparison=None,
                comparison_runs=0, prompt_comparison=None)


def fsd_walkthrough(details=None):
    """A schematic with counts from the actual result, separate from its masks."""
    if details and details.get('method')=='dense':
        return '<div class="dense-walkthrough"><span>DENSE SAM / NATIVE PATH</span><h2>Complete every request.</h2><p>Image encoding → independent point prompts → full native decoding → SAM filters &amp; NMS</p><b>'+str(int(details['native_points']))+' / '+str(int(details['total_points']))+' completed requests</b><p>Dense SAM completes every sampled prompt. It does not use FSD selection.</p></div>'
    query = {}
    if details:
        query = {key: int(details[field]) for key, field in
                 [('total','total_points'),('native','native_points'),('guard','local_guard_points'),('masks','masks')]
                 if field in details}
    url = ANIMATION_URL
    if query:
        url += '?' + urlencode(query)
    return '<iframe class="fsd-embedded-map" title="Interactive FSD-SAM decoding walkthrough" src="'+html.escape(url,quote=True)+'" loading="lazy"></iframe>'


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
    return 60 if int(grid)==32 or large_image else 30


@gpu(duration=comparison_duration)
@torch.inference_mode()
def compute_comparison(state, grid):
    return compare_everything(TEMPLATES['vith'], state['image'], int(grid),
                              run_index=state.get('comparison_runs', 0))


@gpu(duration=15)
@torch.inference_mode()
def compute_prompt_comparison(state, name):
    return compare_prompts(TEMPLATES[name], name, state['image'],
                           state['curated_trajectory'], state['truth'])


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
        return None,None,None,None
    return tuple(prompt_overlay(state,row,field,opacity) for row in (pair['stages'][0],pair['stages'][-1]) for field in ('original','refined'))


def prompt_metrics(pair=None):
    if pair is None:
        return '<div class="prompt-result-intro">Press Run both models to compare the initial decoder output and the corrected output.</div>'
    rows = []
    for stage,row in enumerate(pair['stages']):
        original,refined = row['original']['iou_percent'],row['refined']['iou_percent']
        label = 'Initial prompt' if stage==0 else f'+{stage} foreground correction'+('s' if stage>1 else '')
        rows.append(f'<tr><td>{label}</td><td>{original:.2f}%</td><td><b>{refined:.2f}%</b></td><td>{refined-original:+.2f} pp</td></tr>')
    return '<div class="prompt-score-table"><table><thead><tr><th>Prompt stage</th><th>Original</th><th>Refined</th><th>Difference</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table><p>True target IoU · same prompts · independent previous-mask feedback. Curated COCO example; aggregate results are on the project page.</p></div>'


def run_workspace(state, model, mode, grid, opacity):
    if state['image'] is None:
        raise gr.Error('Choose an image first.')
    try:
        if mode!='Everything':
            if not state.get('curated_trajectory') or state.get('curated_model')!=model.lower() or state.get('curated_mode')!=mode.lower():
                raise gr.Error('Select a curated point/box example first.')
            pair = compute_prompt_comparison(state,model.lower())
            state['prompt_comparison'] = pair
            views = prompt_views(state,opacity)
            name = model
            stages = len(pair['stages'])-1
            details = dict(pair,stages=[dict(row,original={k:v for k,v in row['original'].items() if k!='mask'},refined={k:v for k,v in row['refined'].items() if k!='mask'}) for row in pair['stages']],example=state['example_id'])
            status = f'**{state["example_title"]} · {name}.** Both paths finished: initial prompt and **{stages} foreground corrections**. IoU is measured against the curated target annotation.'
            LOG.info('prompt comparison example=%s model=%s mode=%s iou=%s',state['example_id'],model,mode,[(r['original']['iou_percent'],r['refined']['iou_percent']) for r in pair['stages']])
            label = lambda field,stage: f'{name} / '+('Original' if field=='original' else 'Refined')+f' · {pair["stages"][stage][field]["iou_percent"]:.2f}% IoU'
            return (state,prompt_view(state),gr.update(value=views[3],label=label('refined',-1)),status,
                metrics(model,'One target',f'{len(pair["stages"])} stages','True IoU'),[],gr.update(choices=['All instances'],value='All instances'),fsd_walkthrough(),None,None,pair_metrics(),details,
                gr.update(value=views[0],label=label('original',0)),gr.update(value=views[1],label=label('refined',0)),gr.update(value=views[2],label=label('original',-1)),prompt_metrics(pair))
        pair = compute_comparison(state,grid)
        state['comparison'],state['comparison_runs'] = pair,state.get('comparison_runs',0)+1
        right = pair['results']['fsd']
        state['instances'],state['details'] = right['masks'],right['info']
        scores = [[f'Instance {i+1:02d}',m['area'],round(m['predicted_iou'],3),round(m['stability_score'],3)] for i,m in enumerate(state['instances'])]
        choices = ['All instances']+[r[0] for r in scores]
        left,right_view = comparison_views(state,opacity)
        dense_ms,fsd_ms = [pair['results'][key]['total_ms'] for key in ('dense','fsd')]
        status = f'**Both models finished.** SAM ViT-H: **{dense_ms:,.1f} ms** · FSD-SAM: **{fsd_ms:,.1f} ms**. Same image and {int(grid)} × {int(grid)} grid.'
        LOG.info('paired result device=%s dense_ms=%.3f fsd_ms=%.3f',pair['device'],dense_ms,fsd_ms)
        return (state,grid_view(state,grid),None,status,metrics('ViT-H vs FSD',len(scores),f'{int(grid)} × {int(grid)}','Measured / ms'),scores,
            gr.update(choices=choices,value='All instances'),fsd_walkthrough(state['details']),left,right_view,pair_metrics(pair),pair,None,None,None,prompt_metrics())
    except Exception as error:
        LOG.exception('comparison failed')
        if isinstance(error,gr.Error):
            raise
        raise gr.Error(str(error)[:300]) from error


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
    header = '<div class="studio-header"><div><span class="studio-eyebrow">RESEARCH STUDIO</span><h1>'+title+'</h1><p>One button compares baseline and refinement. See initial and corrected masks, or generate everything with two measured runtimes.</p><div class="studio-links"><a href="https://github.com/thirteen7/Rethinking-Lightweight-SAM" target="_blank" rel="noopener">Code ↗</a><a href="https://thirteen7.github.io/Rethinking-Lightweight-SAM/" target="_blank" rel="noopener">Project page ↗</a></div></div><div class="studio-summary"><span class="studio-summary-label">ONE IMAGE / TWO PATHS</span><div><b>Point &amp; Box</b><span>Original → refined IoU</span></div><div><b>Everything</b><span>ViT-H → FSD-SAM / ms</span></div><p>Curated prompt comparisons. Live automatic segmentation.</p></div></div>'
    source = urlsplit(ANIMATION_URL)
    origin = json.dumps(source.scheme+'://'+source.netloc)
    js = """() => {window.addEventListener('message',event=>{if(event.origin!==ORIGIN || event.data?.type!=='fsd-animation-size')return;const height=Number(event.data.height);if(!Number.isFinite(height)||height<250||height>1800)return;document.querySelectorAll('iframe.fsd-embedded-map').forEach(frame=>{if(event.source===frame.contentWindow)frame.style.setProperty('height',height+'px','important');});});}""".replace('ORIGIN',origin)
    with gr.Blocks(title=title,css=css,js=js,theme=gr.themes.Base(primary_hue='violet',neutral_hue='slate'),delete_cache=(900,900)) as demo:
        gr.HTML(header)
        with gr.Tabs(elem_id='studio-tabs'):
            with gr.Tab('Live studio'):
                session = gr.State(default_state,time_to_live=1200)
                with gr.Row(elem_id='task-bar'):
                    mode = gr.Radio(['Point','Box','Everything'],value='Everything',label='Segmentation mode',elem_id='mode-switch')
                    model = gr.Dropdown(['TinySAM','MobileSAM'],value='TinySAM',label='Lightweight backbone',visible=False,elem_id='model-select')
                stats = gr.HTML(metrics('ViT-H vs FSD','—','16 × 16','Ready'),elem_id='metric-cards')
                with gr.Row(equal_height=False,elem_id='workspace-row'):
                    with gr.Column(scale=1,min_width=220,elem_id='settings-card'):
                        gr.Markdown('### Inference controls')
                        with gr.Group(elem_id='every-controls') as every_controls:
                            grid = gr.Radio([8,16,32],value=16,label='Shared points per side')
                            gr.Markdown('**Left: SAM ViT-H**\n\n**Right: FSD-SAM**\n\nOne button runs both on this image.',elem_id='every-note')
                        with gr.Group(visible=False,elem_id='point-controls') as point_controls:
                            gr.Markdown('**Curated foreground prompts**\n\nThe initial point or box and two positive corrections are already set.\n\nRun once to compare the initial and corrected outputs of both models.',elem_id='point-note')
                        opacity = gr.Slider(0,1,value=.48,step=.01,label='Overlay opacity')
                        button = gr.Button('Run both models →',variant='primary',elem_id='run-segmentation')
                        reset_button = gr.Button('Reset comparison',elem_id='reset-target')
                        gr.HTML('<p class="studio-note">Point / Box: curated annotated targets.<br>Everything: upload or use an example.<br>Frozen weights · FP32.</p>')
                    with gr.Column(scale=3,min_width=300,elem_id='canvas-card'):
                        image = gr.Image(value=grid_view(default_state),type='numpy',label='Input image · Everything uploads',sources=['upload'],height=230,elem_id='input-image')
                        with gr.Group(visible=False,elem_id='prompt-pair-results') as prompt_group:
                            gr.Markdown('#### Initial decoder output',elem_id='initial-stage-title')
                            with gr.Row(equal_height=True):
                                initial_left = gr.Image(label='Original · initial prompt',interactive=False,type='numpy',height=300)
                                initial_right = gr.Image(label='Refined · initial prompt',interactive=False,type='numpy',height=300)
                            gr.Markdown('#### After two foreground corrections',elem_id='corrected-stage-title')
                            with gr.Row(equal_height=True):
                                final_left = gr.Image(label='Original · corrected',interactive=False,type='numpy',height=300)
                                output = gr.Image(label='Refined · corrected',interactive=False,type='numpy',height=300)
                            point_scores = gr.HTML(prompt_metrics(),elem_id='prompt-iou-results')
                        with gr.Group(elem_id='every-pair-results') as pair_group:
                            with gr.Row(equal_height=True):
                                pair_left = gr.Image(type='numpy',label='Left / SAM ViT-H · Dense SAM',interactive=False,height=460)
                                pair_right = gr.Image(type='numpy',label='Right / FSD-SAM · ViT-H',interactive=False,height=460)
                            times = gr.HTML(pair_metrics(),elem_id='pair-timing')
                        pair_details = gr.JSON(visible=False,elem_id='comparison-details')
                        status = gr.Markdown('**Orange bowl is ready.** Press **Run both models** for both masks and measured times.',elem_id='studio-status')
                        with gr.Accordion('FSD-SAM instance details',open=False,elem_id='instance-inspector') as inspector:
                            selected = gr.Dropdown(['All instances'],value='All instances',label='Inspect FSD instance')
                            table = gr.Dataframe(headers=['Instance','Area (px)','SAM predicted IoU','Stability'],datatype=['str','number','number','number'],interactive=False,wrap=True)
                with gr.Group(visible=False,elem_id='curated-gallery') as curated_gallery:
                    gallery = gr.Gallery([(str(project/'site'/row['image']),row['title']+' · '+row['demo_model']+' / '+row['demo_prompt']) for row in gallery_rows],label='Curated comparisons · select a thumbnail',columns=4,height=220,object_fit='cover',allow_preview=False)
                with gr.Group(elem_id='everything-gallery') as every_gallery:
                    every_example_input = gr.Image(visible=False,type='numpy')
                    every_examples = gr.Examples(examples=[str(project/'site/assets/examples/fruit.jpg'),str(project/'site/assets/examples/bear.jpg')],inputs=[every_example_input],label='Everything examples')
            with gr.Tab('How FSD works'):
                diagram = gr.HTML(fsd_walkthrough(),elem_id='fsd-workspace')
        gr.HTML('<div class="studio-footer"><b>'+title+'</b><a href="https://github.com/thirteen7/Rethinking-Lightweight-SAM/blob/main/docs/results.md" target="_blank" rel="noopener">Paper results ↗</a></div>')
        outputs = [session,image,output,status,stats,table,selected,diagram,pair_left,pair_right,times,pair_details,initial_left,initial_right,final_left,point_scores]
        visibility = [every_controls,point_controls,inspector,prompt_group,pair_group,model,curated_gallery,every_gallery]

        def ready(state,value,grid_size):
            view = grid_view(state,grid_size) if value=='Everything' else prompt_view(state)
            name = {'tinysam':'TinySAM','mobilesam':'MobileSAM'}.get(state.get('curated_model'),'Model')
            message = '**Image ready.** Run both models to measure the two automatic generation paths.' if value=='Everything' else f'**{state["example_title"]} is ready.** Initial prompt and two foreground corrections are fixed. Run once for both lightweight models and every stage.'
            return (state,view,gr.update(value=None,label=name+' / Refined · corrected'),message,
                metrics('ViT-H vs FSD' if value=='Everything' else name,'—',f'{int(grid_size)} × {int(grid_size)}' if value=='Everything' else 'Initial + correction','Ready'),
                [],gr.update(choices=['All instances'],value='All instances'),fsd_walkthrough(),None,None,pair_metrics(),None,
                gr.update(value=None,label=name+' / Original · initial prompt'),gr.update(value=None,label=name+' / Refined · initial prompt'),
                gr.update(value=None,label=name+' / Original · corrected'),prompt_metrics())

        def load_image(value,current_mode='Everything',grid_size=16):
            if current_mode!='Everything':
                raise gr.Error('Point and Box use the curated annotated samples. Uploads are available in Everything.')
            if value is None:
                return ready(empty_state(),'Everything',grid_size)
            photo = Image.fromarray(np.asarray(value,dtype=np.uint8)).convert('RGB')
            photo.thumbnail((1800,1800))
            return ready(empty_state(np.asarray(photo).copy()),'Everything',grid_size)

        def switch(state,value,backbone='TinySAM',grid_size=16):
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
            result = list(ready(state,value,grid_size))
            result[1] = gr.update(value=result[1],interactive=enabled,height=230,label='Input image · Everything uploads' if enabled else 'Curated image · fixed foreground prompts')
            return (*result,*[gr.update(visible=show) for show in [enabled,not enabled,enabled,not enabled,enabled,not enabled,not enabled,enabled]])

        def reset_workspace(state,current_mode,backbone,grid_size):
            return switch(state,current_mode,backbone,grid_size)

        image.upload(load_image,inputs=[image,mode,grid],outputs=outputs,api_name='load_image')
        image.clear(lambda: load_image(None),outputs=outputs,api_name=False)
        mode.change(switch,inputs=[session,mode,model,grid],outputs=outputs+visibility,api_name='switch_mode')
        model.change(switch,inputs=[session,mode,model,grid],outputs=outputs+visibility,api_name=False)
        reset_button.click(reset_workspace,inputs=[session,mode,model,grid],outputs=outputs+visibility,api_name='reset_image')
        grid.change(reset_workspace,inputs=[session,mode,model,grid],outputs=outputs+visibility,api_name=False)
        button.click(run_workspace,inputs=[session,model,mode,grid,opacity],outputs=outputs,api_name='segment',concurrency_limit=1)

        def choose_example(state,grid_size,event:gr.SelectData):
            row = gallery_rows[int(event.index)]
            new_mode,new_model = row['demo_prompt'].title(),row['demo_model']
            return (*switch(state,new_mode,new_model,grid_size),gr.update(value=new_mode),gr.update(value=new_model))
        gallery.select(choose_example,inputs=[session,grid],outputs=outputs+visibility+[mode,model],api_name=False)
        every_example_input.change(lambda value,size: load_image(value,'Everything',size),inputs=[every_example_input,grid],outputs=outputs,api_name=False)

        def opacity_changed(state,alpha,selection):
            left,right = comparison_views(state,alpha)
            initial_a,initial_b,final_a,final_b = prompt_views(state,alpha)
            return final_b,left,right,initial_a,initial_b,final_a
        opacity.change(opacity_changed,inputs=[session,opacity,selected],outputs=[output,pair_left,pair_right,initial_left,initial_right,final_left],queue=False,api_name=False)
        selected.change(lambda state,alpha,selection: output_view(state,alpha,selection),inputs=[session,opacity,selected],outputs=[pair_right],queue=False,api_name=False)
    return demo.queue(default_concurrency_limit=1,max_size=12)
