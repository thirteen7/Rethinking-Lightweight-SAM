# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
import torch
from scripts.flip_study.box_decoder_current_anchor import choose_anchor, pair_features
from scripts.flip_study.box_region_evidence import RegionSelector, pair_prediction, region_features
from scripts.flip_study.candidate_selector import choose_candidates, prediction_features
from scripts.flip_study.light_sam_combined import BOX_GATE, choose_point, repair_point
MODELS = ('edgesam', 'vith', 'tinysam', 'mobilesam')
FORMAT = 'backbone_edit_quality_v1'

def canonical_candidates(name, low, quality, tokens):
    """Keep TinySAM's official three candidates plus a duplicate fourth slot."""
    if name == 'tinysam':
        order = quality[:, 1:].argsort(-1, descending=True) + 1
        order = torch.cat([order, order[:, :1]], 1)
        rows = torch.arange(len(low), device=low.device)[:, None]
        return (low[rows, order], quality[rows, order], tokens[rows, order])
    return (low, quality, tokens)

def decode_independent(sam, decoder, pe, name, embedding, boxes, clicks, labels):
    (sparse, dense) = sam.prompt_encoder(points=(clicks, labels) if clicks.shape[1] else None, boxes=boxes, masks=None)
    captured = []
    handle = decoder.transformer.register_forward_hook(lambda _m, _i, out: captured.append(out[0][:, 1:5]))
    try:
        (low, raw) = decoder.predict_masks(image_embeddings=embedding, image_pe=pe, sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('Independent decoder did not produce four mask tokens')
    return canonical_candidates(name, low, raw, captured[0])

class BackboneEditModel:

    def decode_new(self, embedding, boxes, clicks, labels):
        return decode_independent(self.sam, self.decoder, self.light.engine.pe, self.name, embedding, boxes, clicks, labels)

    def box_candidates(self, embedding, boxes, clicks, labels, hw):
        n = len(boxes)
        (old_low, old_raw, old_tokens) = self.light.engine.decode(embedding, clicks if clicks.shape[1] else None, labels if labels.shape[1] else None, boxes)
        feature_coords = clicks if clicks.shape[1] else boxes.reshape(n, 2, 2).mean(1)[:, None]
        feature_labels = labels if labels.shape[1] else boxes.new_full((n, 1), -1.0)
        (stats, compatible) = prediction_features(old_low, old_raw, feature_coords, feature_labels, hw, clicks.shape[1] + 1)
        old_choice = choose_candidates(self.light.box(old_tokens, stats), compatible, BOX_GATE)
        rows = torch.arange(n, device=boxes.device)
        current = old_low[rows, old_choice, None]
        (new_low, new_raw, new_tokens) = self.decode_new(embedding, boxes, clicks, labels)
        packed = pair_features(current, old_raw[rows, old_choice, None], old_tokens[rows, old_choice, None], new_low, new_raw, new_tokens, boxes, clicks, labels, hw)
        all_low = torch.cat([current, new_low], 1)
        region = region_features(embedding, all_low, hw, 'edit')
        region_packed = torch.stack([region[:, :3], region[:, 3:4].expand(-1, 3, -1)], 1)
        (tokens, pair_stats, geometry, compatible) = packed
        features = (tokens, torch.cat([pair_stats, region_packed], -1), geometry, compatible)
        groups = [dict(low=old_low, raw=old_raw, tokens=old_tokens), dict(low=new_low, raw=new_raw, tokens=new_tokens)]
        return (all_low, features, old_choice, groups)

    @torch.inference_mode()
    def predict(self, embedding, image, coords, labels, hw, nh, prompt, boxes=None, previous=None):
        if prompt == 'point':
            if boxes is not None or (coords.shape[1] > 1 and previous is None):
                raise ValueError('Point prompts require previous selected logits after the first click')
            bundle = self.light.engine.point_bundle(embedding, image, coords, labels, hw, previous, repair=self.light.residual is not None)
            head = self.light.first if coords.shape[1] == 1 else self.light.feedback
            (selected, choice) = choose_point(head, bundle)
            if previous is not None:
                selected = repair_point(self.light.residual, bundle, image, coords, labels, hw, previous, selected, self.light.residual_scale)
            groups = [dict(low=bundle['low'], raw=bundle['quality'], tokens=bundle['tokens'])]
            old_choice = None
        elif prompt == 'box':
            if self.comparator is None:
                raise ValueError('Train the edit comparator before box prediction')
            if boxes is None or coords.shape[1] > 2:
                raise ValueError('Box prompts require boxes and at most two correction clicks')
            (all_low, features, old_choice, groups) = self.box_candidates(embedding, boxes, coords, labels, hw)
            choice = choose_anchor(pair_prediction(self.comparator, *features[:3]), features[3])
            selected = all_low[torch.arange(len(choice), device=choice.device), choice, None]
        else:
            raise ValueError(prompt)
        masks = self.sam.postprocess_masks(selected, hw, nh)[:, 0] > 0
        reference = boxes if prompt == 'box' else selected.new_tensor([0.0, 0.0, float(hw[1]), float(hw[0])])[None].expand(len(selected), -1)
        trace = dict(prompt=prompt, groups=groups, choice=choice, old_choice=old_choice)
        self.last_trace = trace
        return (masks, choice, selected)
