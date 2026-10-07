# Extracted from the verified research implementation; see LICENSE and NOTICE.
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.flip_study.candidate_selector import GainSelector, choose_candidates, prediction_features
from scripts.flip_study.click_residual import ClickResidualHead, crop, geometry, make_inputs, paste
from scripts.flip_study.feedback_adaptation import TemporalSelector, latest_click, point_violations, temporal_features
from scripts.flip_study.light_sam_adapter import FourCandidateEngine, load_sam
from scripts.flip_study.local_attention_selector import LocalSelector, local_grid
POINT_GATE = dict(min_gain_pp=0.0, max_harm_probability=0.5)
BOX_GATE = dict(min_gain_pp=0.0, max_harm_probability=0.2)

class LightFeatureEngine(FourCandidateEngine):
    """Capture the same decoder map used by the QA-SAM attention and repair heads."""

    @torch.inference_mode()
    def point_bundle(self, embedding, image, coords, labels, hw, previous=None, repair=False):
        (first_grid, first_valid, first_geo) = local_grid(coords, hw)
        (latest, _) = latest_click(coords, labels)
        (last_grid, last_valid, last_geo) = local_grid(latest, hw)
        region = geometry(coords, labels, hw) if repair and previous is not None else None
        sampled = []

        def capture(_module, _args, features):
            first = F.grid_sample(features, first_grid, align_corners=False)[:, :, :, 0].transpose(1, 2)
            last = F.grid_sample(features, last_grid, align_corners=False)[:, :, :, 0].transpose(1, 2)
            sampled.append((first, last, crop(features, region) if region is not None else None))
        handle = self.sam.mask_decoder.output_upscaling.register_forward_hook(capture)
        try:
            (low, quality, tokens) = self.decode(embedding, coords, labels, previous=previous)
        finally:
            handle.remove()
        if len(sampled) != 1:
            raise RuntimeError('Expected one decoder feature map')
        (first, last, patch) = sampled[0]
        if first.shape[-1] != 32:
            raise RuntimeError(f'Unexpected decoder channels: {first.shape}')

        def context(grid, feature, geo, valid):
            rgb = F.grid_sample(image.expand(len(coords), -1, -1, -1), grid, align_corners=False)[:, :, :, 0].transpose(1, 2)
            logits = F.grid_sample(low, grid, align_corners=False)[:, :, :, 0].transpose(1, 2).clamp(-10, 10) / 10
            return torch.cat([feature, rgb, logits, geo], -1).masked_fill(~valid[..., None], 0.0)
        (stats, compatible) = prediction_features(low, quality, coords, labels, hw, coords.shape[1])
        return dict(low=low, quality=quality, tokens=tokens, stats=stats, compatible=compatible, context=context(first_grid, first, first_geo, first_valid), valid=first_valid, latest_context=context(last_grid, last, last_geo, last_valid), latest_valid=last_valid, temporal=temporal_features(low, previous, coords, labels, hw) if previous is not None else None, patch_features=patch, region=region)

def score_point(head, bundle):
    args = [bundle[k].float() if k != 'valid' else bundle[k] for k in ('tokens', 'stats', 'context', 'valid')]
    if isinstance(head, TemporalSelector):
        if bundle['temporal'] is None:
            raise ValueError('Temporal head requires preceding mask feedback')
        args += [bundle['latest_context'].float(), bundle['latest_valid'], bundle['temporal'].float()]
    return head(*args)

def choose_point(head, bundle):
    prediction = score_point(head, bundle)
    choice = choose_candidates(prediction, bundle['compatible'], POINT_GATE)
    selected = bundle['low'][torch.arange(len(choice), device=choice.device), choice][:, None]
    return (selected, choice)

def repair_point(head, bundle, image, coords, labels, hw, previous, selected, scale):
    if head is None or scale == 0:
        return selected
    (inputs, _, support) = make_inputs(bundle['patch_features'], image, selected, previous, coords, labels, hw, bundle['region'])
    proposed = paste(selected, head(inputs, support) * scale, bundle['region'])
    introduced = point_violations(proposed, coords, labels) & ~point_violations(selected, coords, labels)
    accept = ~introduced.flatten(1).any(1)
    return torch.where(accept[:, None, None, None], proposed, selected)

def _load(path, model, component, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('model') != model or state.get('component') != component or state.get('format_version') != 2:
        raise ValueError(f'Wrong {component} checkpoint for {model}: {path}')
    return state['state_dict']

class LightCombinedModel:

    def __init__(self, name, output, device='cuda', with_residual=True):
        output = Path(output) / name
        self.sam = load_sam(name, device)
        self.engine = LightFeatureEngine(self.sam, name)
        self.first = LocalSelector('local_attention').to(device).eval()
        self.first.load_state_dict(_load(output / 'point_attention.pth', name, 'point_attention', device), strict=True)
        self.feedback = TemporalSelector().to(device).eval()
        self.feedback.load_state_dict(_load(output / 'point_feedback.pth', name, 'point_feedback', device), strict=True)
        self.box = GainSelector().to(device).eval()
        self.box.load_state_dict(_load(output / 'box_selector.pth', name, 'box_selector', device), strict=True)
        self.residual = None
        self.residual_scale = 0.0
        if with_residual:
            residual_state = torch.load(output / 'point_residual.pth', map_location='cpu', weights_only=True)
            if (residual_state.get('format_version'), residual_state.get('model'), residual_state.get('component')) != (2, name, 'point_residual'):
                raise ValueError(f'Wrong point residual checkpoint for {name}')
            self.residual = ClickResidualHead().to(device).eval()
            self.residual.load_state_dict(residual_state['state_dict'], strict=True)
            self.residual_scale = float(residual_state['scale'])
            if self.residual_scale not in (0.0, 0.25, 0.5, 1.0):
                raise ValueError(f'Invalid residual scale: {self.residual_scale}')

    @torch.inference_mode()
    def point(self, embedding, image, coords, labels, input_hw, native_hw, previous=None):
        if coords.shape[1] > 1 and previous is None:
            raise ValueError('Later point rounds require the previous selected logits')
        bundle = self.engine.point_bundle(embedding, image, coords, labels, input_hw, previous, repair=self.residual is not None)
        head = self.first if coords.shape[1] == 1 else self.feedback
        (low, choice) = choose_point(head, bundle)
        if previous is not None:
            low = repair_point(self.residual, bundle, image, coords, labels, input_hw, previous, low, self.residual_scale)
        mask = self.sam.postprocess_masks(low, input_hw, native_hw)[:, 0] > 0
        return (mask, choice, low)

    @torch.inference_mode()
    def box_prompt(self, embedding, boxes, coords, labels, input_hw, native_hw):
        (low, quality, tokens) = self.engine.decode(embedding, coords if coords.shape[1] else None, labels if labels.shape[1] else None, boxes)
        if coords.shape[1]:
            (feature_coords, feature_labels) = (coords, labels)
        else:
            feature_coords = boxes.reshape(len(boxes), 2, 2).mean(1)[:, None]
            feature_labels = torch.full((len(boxes), 1), -1.0, device=boxes.device)
        (stats, compatible) = prediction_features(low, quality, feature_coords, feature_labels, input_hw, coords.shape[1] + 1)
        choice = choose_candidates(self.box(tokens, stats), compatible, BOX_GATE)
        selected = low[torch.arange(len(choice), device=choice.device), choice][:, None]
        mask = self.sam.postprocess_masks(selected, input_hw, native_hw)[:, 0] > 0
        return (mask, choice, selected)
