# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import torch
CHECKPOINTS = {'mobilesam': ROOT / '.local-deps/MobileSAM/weights/mobile_sam.pt', 'tinysam': ROOT / 'weights/tinysam.pth', 'edgesam': ROOT / 'weights/edge_sam.pth', 'vith': ROOT / 'weights/sam_vit_h_4b8939.pth'}

def load_sam(name: str, device: str='cuda'):
    if name not in CHECKPOINTS:
        raise ValueError(f'Unknown model {name}')
    checkpoint = CHECKPOINTS[name]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if name == 'mobilesam':
        from mobile_sam import sam_model_registry
        model = sam_model_registry['vit_t']()
    elif name == 'tinysam':
        from tinysam import sam_model_registry
        model = sam_model_registry['vit_t']()
    elif name == 'edgesam':
        from edge_sam import sam_model_registry
        model = sam_model_registry['edge_sam']()
    else:
        from qa_sam import sam_model_registry
        model = sam_model_registry['vit_h']()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)

class FourCandidateEngine:

    def __init__(self, sam, name):
        self.sam = sam
        self.name = name
        self.pe = sam.prompt_encoder.get_dense_pe()
        self.tokens = None

    def _capture(self, module, inputs, output):
        self.tokens = output[0][:, 1:5].detach()

    @torch.inference_mode()
    def decode(self, embedding, coords=None, labels=None, boxes=None, previous=None):
        n = len(boxes) if boxes is not None else len(coords)
        points = (coords, labels) if coords is not None else None
        (sparse, dense) = self.sam.prompt_encoder(points=points, boxes=boxes, masks=previous)
        handle = self.sam.mask_decoder.transformer.register_forward_hook(self._capture)
        try:
            kwargs = dict(image_embeddings=embedding, image_pe=self.pe, sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
            (low, quality) = self.sam.mask_decoder.predict_masks(**kwargs)
        finally:
            handle.remove()
        tokens = self.tokens
        self.tokens = None
        if self.name == 'tinysam':
            order = quality[:, 1:].argsort(-1, descending=True) + 1
            order = torch.cat([order, order[:, :1]], dim=1)
            row = torch.arange(n, device=low.device)[:, None]
            (low, quality, tokens) = (low[row, order], quality[row, order], tokens[row, order])
        if low.shape != (n, 4, 256, 256) or quality.shape != (n, 4) or tokens.shape != (n, 4, 256):
            raise RuntimeError(f'Unexpected four-candidate shapes: {low.shape}, {quality.shape}, {tokens.shape}')
        return (low, quality, tokens)

    @torch.inference_mode()
    def native(self, low, input_hw, native_hw):
        return self.sam.postprocess_masks(low, input_hw, native_hw) > 0
