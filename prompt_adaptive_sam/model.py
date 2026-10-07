"""One complete checkpoint per backbone; no separate module files at inference."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from ._bootstrap import bootstrap

bootstrap()
import torch
from scripts.side_coco_eval.backbone_edit_model import BackboneEditModel, RegionSelector
from scripts.flip_study import light_sam_combined as light
from scripts.side_coco_eval.prompt_contract import prompt_contract

FORMAT = 'prompt_adaptive_sam_v1'
MODELS = ('tinysam', 'mobilesam', 'vith')
COMPONENTS = ('point_attention','point_feedback','point_residual','box_selector',
              'independent_decoder','edit_comparator')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): value.update(block)
    return value.hexdigest()


def tensor_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor): raise TypeError(name)
        tensor = tensor.detach().cpu().contiguous()
        header = json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode()
        digest.update(len(header).to_bytes(8,'little')); digest.update(header)
        digest.update(memoryview(tensor.numpy()).cast('B'))
    return digest.hexdigest()


def load_payload(checkpoint):
    value = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if value.get('format') != FORMAT or value.get('model') not in MODELS:
        raise ValueError('Expected a released prompt-adaptive model checkpoint')
    if value.get('added_quality_head') is not False or value.get('rescoring') is not False:
        raise ValueError('This implementation has no additional quality or rescoring head')
    if value.get('prompt_contract') != prompt_contract():
        raise ValueError('Training/inference prompt encoding or source fingerprint differs')
    if value.get('complete_training') is not True or value.get('smoke') is not False:
        raise ValueError('Incomplete training checkpoint')
    return value


class PromptAdaptiveModel(BackboneEditModel):
    def predict_scored(self,*args,**kwargs):
        raise RuntimeError('Use predict(); external detector scores remain unchanged')


def from_payload(entry,device='cpu',verify=True):
    name = entry['model']
    if name not in MODELS: raise ValueError(name)
    if set(entry['state_dicts']) != {'sam',*COMPONENTS}:
        raise ValueError('Expected the backbone and all six refinement modules')
    if verify:
        for component,state in entry['state_dicts'].items():
            if tensor_sha256(state) != entry['tensor_sha256'][component]:
                raise ValueError('Tensor fingerprint mismatch: '+component)
            if any(t.is_floating_point() and not torch.isfinite(t).all() for t in state.values()):
                raise ValueError('Non-finite parameters: '+component)
    if name == 'mobilesam':
        from mobile_sam import sam_model_registry
        sam = sam_model_registry['vit_t']()
    elif name == 'tinysam':
        from tinysam import sam_model_registry
        sam = sam_model_registry['vit_t']()
    else:
        from qa_sam import sam_model_registry
        sam = sam_model_registry['vit_h']()
    sam.load_state_dict(entry['state_dicts']['sam'],strict=True)
    sam = sam.to(device).eval().requires_grad_(False)
    combined = light.LightCombinedModel.__new__(light.LightCombinedModel)
    combined.sam = sam
    combined.engine = light.LightFeatureEngine(sam,name)
    modules = {'first':(light.LocalSelector('local_attention'),'point_attention'),
               'feedback':(light.TemporalSelector(),'point_feedback'),
               'residual':(light.ClickResidualHead(),'point_residual'),
               'box':(light.GainSelector(),'box_selector')}
    for attr,(module,component) in modules.items():
        module.load_state_dict(entry['state_dicts'][component],strict=True)
        setattr(combined,attr,module.to(device).eval().requires_grad_(False))
    combined.residual_scale = float(entry['point_residual_scale'])
    if combined.residual_scale not in (0.,.25,.5,1.): raise ValueError('Invalid residual scale')
    model = PromptAdaptiveModel.__new__(PromptAdaptiveModel)
    model.name = name
    model.light = combined
    model.sam = sam
    model.decoder = copy.deepcopy(sam.mask_decoder)
    model.decoder.load_state_dict(entry['state_dicts']['independent_decoder'],strict=True)
    model.decoder = model.decoder.to(device).eval().requires_grad_(False)
    model.comparator = RegionSelector()
    model.comparator.load_state_dict(entry['state_dicts']['edit_comparator'],strict=True)
    model.comparator = model.comparator.to(device).eval().requires_grad_(False)
    model.quality_heads = None
    model.last_trace = None
    return model


def load_model(checkpoint,device='cpu'):
    """Load one .pth containing the official backbone and all refinement modules."""
    return from_payload(load_payload(Path(checkpoint)),device=device)


def export_components(model_name,base_checkpoint,component_root,destination):
    """Pack completed model-specific training outputs into one portable .pth."""
    from scripts.flip_study.light_sam_adapter import CHECKPOINTS
    if model_name not in MODELS: raise ValueError(model_name)
    folder = Path(component_root)/model_name
    receipt = json.loads((folder/'edit_complete.json').read_text())
    if not receipt.get('complete') or receipt.get('smoke'):
        raise ValueError('Only complete non-smoke training may be exported')
    base_checkpoint = Path(base_checkpoint)
    if receipt['base_sha256'] != sha256(base_checkpoint): raise ValueError('Base changed')
    states = {'sam':torch.load(base_checkpoint,map_location='cpu',weights_only=True)}
    sources = {'sam':sha256(base_checkpoint)}
    scale = None
    for component in COMPONENTS:
        path = folder/(component+'.pth')
        if receipt['components'][component] != sha256(path): raise ValueError(component+' changed')
        payload = torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('model') != model_name or payload.get('smoke'):
            raise ValueError('Wrong or smoke-only component: '+component)
        states[component] = payload['state_dict']; sources[component] = sha256(path)
        if component == 'point_residual': scale = float(payload['scale'])
    payload = dict(format=FORMAT,model=model_name,state_dicts=states,
        tensor_sha256={name:tensor_sha256(state) for name,state in states.items()},
        source_file_sha256=sources,point_residual_scale=scale,complete_training=True,smoke=False,
        added_quality_head=False,rescoring=False,score_policy='detector',
        prompt_contract=prompt_contract(),training_data='SA-1B train only')
    from_payload(payload,device='cpu')  # Strict loading precedes publication.
    destination = Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_suffix('.tmp'); torch.save(payload,temporary); temporary.replace(destination)
    return sha256(destination)
