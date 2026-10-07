"""Shared SegAny / SegEvery research workspace."""
from pathlib import Path
import copy
import html
import json
import logging
import os
import time
from urllib.parse import urlencode

DEVICE = os.environ.get('SAM_DEMO_DEVICE', 'cuda')
if DEVICE == 'cuda':
    import spaces  # Initialize ZeroGPU before torch.
    gpu = spaces.GPU
else:
    def gpu(**unused):
        return lambda fn: fn

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
from prompt_adaptive_sam import Predictor, EverythingGenerator
from prompt_adaptive_sam.everything import render_instances
from download_models import download

LOG = logging.getLogger('segmentation-workspace')
TEMPLATES = {}


def empty_state(image=None):
    return dict(image=image, points=[], box=None, corners=[], previous=None,
                embedding=None, image_tensor=None, model=None, rounds=0,
                previous_count=0, native_hw=None, input_hw=None, mask=None,
                instances=[], details=None, preset=False)


def fsd_walkthrough(details=None):
    """A schematic with counts from the actual result, separate from its masks."""
    if details and details.get('method')=='dense':
        return '<div class="dense-walkthrough"><span>DENSE SAM / NATIVE PATH</span><h2>Complete every request.</h2><p>Image encoding → independent point prompts → full native decoding → SAM filters &amp; NMS</p><b>'+str(int(details['native_points']))+' / '+str(int(details['total_points']))+' completed requests</b><p>Dense SAM completes every sampled prompt. It does not use FSD selection.</p></div>'
    query = {}
    if details:
        query = {key: int(details[field]) for key, field in
                 [('total','total_points'),('native','native_points'),('guard','local_guard_points'),('masks','masks')]
                 if field in details}
    url = 'https://thirteen7.github.io/Rethinking-Lightweight-SAM/assets/fsd-workflow.html'
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
    fields = [('MODE',mode),('INSTANCES',masks),('PROMPTS / ROUND',requests),('INFERENCE',elapsed)]
    return '<div class="metrics-row">'+''.join(
        f'<div class="metric-tile"><span>{label}</span><strong>{html.escape(str(value))}</strong></div>'
        for label,value in fields)+'</div>'


def loaded(image):
    if image is None:
        state,message = empty_state(),'Choose an example or upload a photograph.'
    else:
        image = Image.fromarray(np.asarray(image,dtype=np.uint8)).convert('RGB')
        image.thumbnail((1800,1800))
        state = empty_state(np.asarray(image).copy())
        message = 'Image ready. Select a target, or choose Everything for automatic generation.'
    return state,prompt_view(state),None,message,metrics(),[],gr.update(choices=['All instances'],value='All instances'),fsd_walkthrough()


def reset(state):
    result = loaded(state.get('image'))
    result[0]['preset'] = state.get('preset',False)
    return result


def clear():
    state,*rest = loaded(None)
    return state,None,rest[2],rest[3],rest[4],rest[5],rest[6]


def select_point(state,mode,label,event:gr.SelectData):
    if mode=='Everything':
        return state,prompt_view(state),'Everything generates its own grid. Run segmentation to begin.'
    if state['image'] is None:
        raise gr.Error('Upload an image first.')
    if state['rounds']>=3:
        raise gr.Error('Three rounds are complete. Reset to select another target.')
    x,y = map(float,event.index)
    h,w = state['image'].shape[:2]
    if not (0<=x<w and 0<=y<h):
        raise gr.Error('Choose a point inside the image.')
    if mode=='Box' and state['rounds']==0:
        if state['box'] is not None:
            raise gr.Error('Run the box prediction before adding a correction.')
        state['corners'].append([x,y])
        if len(state['corners'])==2:
            a,b = state['corners']
            box = [min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])]
            state['corners'] = []
            if box[2]-box[0]<3 or box[3]-box[1]<3:
                raise gr.Error('Choose two distinct corners around the target.')
            state['box'],message = box,'Box ready. Run segmentation.'
        else:
            message = 'Select the opposite corner of the box.'
    else:
        if len(state['points'])>state['previous_count']:
            raise gr.Error('Run the current prompt before adding another correction.')
        positive = label=='Foreground (+)'
        if mode=='Point' and state['rounds']==0 and not positive:
            raise gr.Error('Begin with a foreground click.')
        state['points'].append(dict(x=x,y=y,label=int(positive)))
        message = 'Prompt ready. Run segmentation.'
    return state,prompt_view(state),message


@gpu(duration=30)
@torch.inference_mode()
def compute(state,name,mode,method,grid):
    """Return NumPy arrays and RLE across the GPU boundary."""
    started = time.perf_counter()
    predictor = copy.copy(TEMPLATES[name])
    if state['embedding'] is None or state['model']!=name:
        predictor.set_image(state['image'])
    else:
        predictor.embedding = torch.as_tensor(state['embedding'],device=DEVICE)
        predictor.image = torch.as_tensor(state['image_tensor'],device=DEVICE)
        predictor.native_hw,predictor.input_hw = state['native_hw'],state['input_hw']
    result = dict(embedding=predictor.embedding.cpu().numpy().copy(),
                  image_tensor=predictor.image.cpu().numpy().copy(),
                  native_hw=predictor.native_hw,input_hw=predictor.input_hw)
    if mode=='Everything':
        masks,details = EverythingGenerator(predictor,method='fsd' if method=='FSD-SAM' else 'dense',
            points_per_side=int(grid),points_per_batch=4 if DEVICE=='cpu' else 32).generate()
        result.update(instances=masks,details=details)
    else:
        coords = [[p['x'],p['y']] for p in state['points']]
        labels = [p['label'] for p in state['points']]
        mask,logits,choice = predictor.predict(coords,labels,
            box=state['box'] if mode=='Box' else None,previous=state['previous'])
        if not np.isfinite(logits).all():
            raise RuntimeError('Non-finite prediction.')
        result.update(mask=mask,previous=logits,choice=choice)
        predictor.model.last_trace = None
    result['seconds'] = time.perf_counter()-started
    return result


def run(state,model,mode,method,grid,opacity):
    if state['image'] is None:
        raise gr.Error('Choose an example or upload an image first.')
    if mode!='Everything':
        if state['rounds']>=3:
            raise gr.Error('Three rounds are complete. Reset prompts for another target.')
        if mode=='Box' and state['box'] is None:
            raise gr.Error('Select two box corners first.')
        if len(state['points'])!=state['rounds']+(1 if mode=='Point' else 0):
            raise gr.Error('Add one corrective click before the next round.')
    name = model.lower()
    LOG.info('prediction model=%s mode=%s grid=%s image_shape=%s',name,mode,grid,state['image'].shape)
    try:
        result = compute(state,name,mode,method,grid)
    except Exception as error:
        LOG.exception('prediction failed')
        raise gr.Error(str(error)[:300]) from error
    for key in ('embedding','image_tensor','native_hw','input_hw'):
        state[key] = result[key]
    state['model'] = name
    if mode=='Everything':
        state['instances'],state['details'] = result['instances'],result['details']
        info = result['details']
        message = f"**{len(state['instances'])} instances** · {int(grid)} × {int(grid)} independent prompts · {info['native_points']} completed requests."
        scores = [[f'Instance {i+1:02d}',m['area'],round(m['predicted_iou'],3),round(m['stability_score'],3)] for i,m in enumerate(state['instances'])]
        summary = metrics(method,len(scores),f'{int(grid)} × {int(grid)}',f"{result['seconds']:.2f} s")
        choices = ['All instances']+[f'Instance {i+1:02d}' for i in range(len(scores))]
    else:
        state['mask'],state['previous'] = result['mask'],result['previous']
        state['rounds'] += 1
        state['previous_count'] = len(state['points'])
        message = f"**Round {state['rounds']} / 3** · Candidate {result['choice']}. "+('Add one corrective click to continue.' if state['rounds']<3 else 'Reset prompts to select another target.')
        summary = metrics(mode,1,f"{state['rounds']} / 3",f"{result['seconds']:.2f} s")
        scores,choices = [],['All instances']
    return state,grid_view(state,grid) if mode=='Everything' else prompt_view(state),output_view(state,opacity),message,summary,scores,gr.update(choices=choices,value='All instances'),fsd_walkthrough(state.get('details'))


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
    css = (project/'demo/space/workspace.css').read_text()
    default_image = np.asarray(Image.open(project/'site/assets/examples/fruit.jpg').convert('RGB'))
    preview = json.loads((project/'site/assets/everything/fruit-tinysam-fsd-16.json').read_text())
    default_state = empty_state(default_image)
    default_state['preset'] = True
    default_state['instances'],default_state['details'] = preview['masks'],preview['info']
    preview_scores = [[f'Instance {i+1:02d}',mask['area'],round(mask['predicted_iou'],3),round(mask['stability_score'],3)]
                      for i,mask in enumerate(preview['masks'])]
    preview_choices = ['All instances']+[f'Instance {i+1:02d}' for i in range(len(preview_scores))]
    header = '''<div class="studio-header"><div><span class="studio-eyebrow">RETHINKING LIGHTWEIGHT SAM / RESEARCH STUDIO</span><h1>Precision, one prompt.<br><em>Coverage, every instance.</em></h1><p>Explore prompt-adaptive refinement and automatic mask generation in one workspace.</p><div class="studio-links"><a href="https://thirteen7.github.io/Rethinking-Lightweight-SAM/" target="_blank">Project ↗</a><a href="https://github.com/thirteen7/Rethinking-Lightweight-SAM" target="_blank">Code &amp; models ↗</a></div></div><div class="studio-summary"><span class="studio-summary-label">TWO INFERENCE PATHS</span><div><b>SegAny</b><span>Point · Box · Correction</span></div><div><b>SegEvery</b><span>FSD-SAM · Dense SAM</span></div><p>One complete checkpoint per backbone.</p></div></div>'''
    with gr.Blocks(title='Rethinking Lightweight SAM · Studio',css=css,
                   theme=gr.themes.Base(primary_hue='violet',neutral_hue='slate'),delete_cache=(900,900)) as demo:
        gr.HTML(header)
        with gr.Tabs(elem_id='studio-tabs'):
            with gr.Tab('Live studio'):
                session = gr.State(default_state,time_to_live=1200)
                with gr.Row(elem_id='task-bar'):
                    mode = gr.Radio(['Point','Box','Everything'],value='Everything',label='Segmentation mode',elem_id='mode-switch')
                    model = gr.Dropdown(['TinySAM','MobileSAM'],value='TinySAM',label='Backbone',elem_id='model-select')
                stats = gr.HTML(metrics('FSD-SAM',len(preview_scores),'16 × 16','Recorded preview'),elem_id='metric-cards')
                with gr.Row(equal_height=False,elem_id='workspace-row'):
                    with gr.Column(scale=1,min_width=235,elem_id='settings-card'):
                        gr.Markdown('### Inference controls')
                        with gr.Group(visible=True,elem_id='every-controls') as every_controls:
                            method = gr.Dropdown(['FSD-SAM','Dense SAM'],value='FSD-SAM',label='Generator')
                            grid = gr.Radio([8,16,32],value=16,label='Points per side')
                            gr.Markdown('Independent positive prompts · three native candidates · fixed SAM quality/stability filters.',elem_id='every-note')
                        with gr.Group(visible=False,elem_id='point-controls') as point_controls:
                            label = gr.Radio(['Foreground (+)','Background (−)'],value='Foreground (+)',label='Correction label')
                            gr.Markdown('Point: one foreground click.\n\nBox: two opposite corners.\n\nThen add one correction per round.',elem_id='point-note')
                        opacity = gr.Slider(0,1,value=.48,step=.01,label='Overlay opacity')
                        button = gr.Button('Run segmentation →',variant='primary',elem_id='run-segmentation')
                        reset_button = gr.Button('Reset target',elem_id='reset-target')
                        gr.HTML('<p class="studio-note">FP32 inference · frozen backbone<br>Uploads use temporary Space storage.<br>Instance colors do not denote semantic classes.</p>')
                    with gr.Column(scale=3,min_width=300,elem_id='canvas-card'):
                        with gr.Group(visible=True,elem_id='result-view-options') as view_group:
                            result_view = gr.Radio(['FSD walkthrough','Instance masks'],value='FSD walkthrough',label='Everything view')
                        with gr.Row():
                            image = gr.Image(value=grid_view(default_state),type='numpy',label='01 / Image & prompts',sources=['upload'],height=580)
                            with gr.Column():
                                with gr.Group(visible=False,elem_id='mask-result-group') as mask_group:
                                    output = gr.Image(value=output_view(default_state),type='numpy',label='02 / Segmentation',interactive=False,height=580)
                                with gr.Group(visible=True,elem_id='flow-result-group') as flow_group:
                                    diagram = gr.HTML(fsd_walkthrough(preview['info']),elem_id='fsd-workspace')
                        status = gr.Markdown('**Recorded CPU example** · Orange bowl · TinySAM · 16 × 16. Run segmentation to generate a new result, or upload a photograph.',elem_id='studio-status')
                        outputs = [session,image,output,status,stats]
                        with gr.Group(visible=True,elem_id='instance-inspector') as inspector:
                            selected = gr.Dropdown(preview_choices,value='All instances',label='Inspect instance')
                            table = gr.Dataframe(value=preview_scores,headers=['Instance','Area (px)','SAM predicted IoU','Stability'],datatype=['str','number','number','number'],interactive=False,label='Instance inventory',wrap=True)
                        full_outputs = outputs+[table,selected,diagram]
                        showcase_records = json.loads((project/'site/examples.json').read_text())
                        showcase_ids = ['fruit']+list(dict.fromkeys(showcase_records['default_examples'][name][prompt] for name in ('tinysam','mobilesam') for prompt in ('point','box')))
                        gr.Examples(examples=[str(project/next('site/'+row['image'] for row in showcase_records['examples'] if row['id']==key)) for key in showcase_ids],inputs=[image],outputs=full_outputs,fn=loaded,cache_examples=False,run_on_click=True,label='Start from an image')
            with gr.Tab('Original vs refined'):
                gr.HTML('<div class="comparison-intro"><span>CURATED / FIRST-ROUND COMPARISONS</span><h2>One prompt. A clearer target.</h2><p>Original TinySAM or MobileSAM versus prompt-adaptive refinement under identical prompts. Scores are measured legacy IoU for each illustrated target.</p></div><iframe class="comparison-explorer-frame" title="Original lightweight SAM versus refined predictions" src="https://thirteen7.github.io/Rethinking-Lightweight-SAM/?embed=showcase" loading="lazy"></iframe>')
        gr.HTML('<div class="studio-footer"><b>Rethinking Lightweight SAM</b><span>Prompt-Adaptive Refinement &amp; Efficient Segment Everything Inference</span><a href="https://github.com/thirteen7/Rethinking-Lightweight-SAM/blob/main/docs/everything.md" target="_blank">Everything protocol ↗</a></div>')
        image.upload(loaded,inputs=[image],outputs=full_outputs)
        image.select(select_point,inputs=[session,mode,label],outputs=[session,image,status])
        image.clear(clear,outputs=[session,output,status,stats,table,selected,diagram])
        reset_button.click(reset,inputs=[session],outputs=full_outputs)
        method.change(reset,inputs=[session],outputs=full_outputs)
        grid.change(reset,inputs=[session],outputs=full_outputs)
        def switch(state,value,backbone='TinySAM',view='FSD walkthrough',grid_size=16):
            if state.get('preset'):
                key = showcase_records['default_examples'][backbone.lower()][value.lower()]
                row = next(row for row in showcase_records['examples'] if row['id']==key)
                state = empty_state(np.asarray(Image.open(project/'site'/row['image']).convert('RGB')))
                state['preset'] = True
            enabled = value=='Everything'
            result = list(reset(state))
            if enabled:
                result[1] = grid_view(state,int(grid_size))
            return (*result,gr.update(visible=enabled),gr.update(visible=not enabled),gr.update(visible=enabled),gr.update(visible=enabled),gr.update(visible=enabled and view=='FSD walkthrough'),gr.update(visible=not enabled or view!='FSD walkthrough'))
        visibility_outputs = [every_controls,point_controls,inspector,view_group,flow_group,mask_group]
        mode.change(switch,inputs=[session,mode,model,result_view,grid],outputs=full_outputs+visibility_outputs)
        def change_model(state,backbone,value,view,grid_size):
            return switch(state,value,backbone,view,grid_size)
        model.change(change_model,inputs=[session,model,mode,result_view,grid],outputs=full_outputs+visibility_outputs)
        def change_result_view(view,value):
            enabled = value=='Everything'
            return gr.update(visible=enabled and view=='FSD walkthrough'),gr.update(visible=not enabled or view!='FSD walkthrough')
        result_view.change(change_result_view,inputs=[result_view,mode],outputs=[flow_group,mask_group],queue=False)
        button.click(run,inputs=[session,model,mode,method,grid,opacity],outputs=full_outputs,api_name='segment',concurrency_limit=1)
        opacity.change(output_view,inputs=[session,opacity,selected],outputs=[output],queue=False)
        selected.change(output_view,inputs=[session,opacity,selected],outputs=[output],queue=False)
    return demo.queue(default_concurrency_limit=1,max_size=12)
