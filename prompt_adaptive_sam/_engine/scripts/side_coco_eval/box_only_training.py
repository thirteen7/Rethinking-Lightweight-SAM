# Extracted from the verified research implementation; see LICENSE and NOTICE.
from pathlib import Path
from types import SimpleNamespace
import copy
import torch
from scripts.flip_study.light_sam_adapter import load_sam, FourCandidateEngine
from scripts.flip_study.candidate_selector import GainSelector
from scripts.side_coco_eval.backbone_edit_model import BackboneEditModel, FORMAT

class BoxOnlyTrainingModel(BackboneEditModel):

    def __init__(self, name, output, *, quality=False, comparator=False, allow_smoke=False):
        if quality or comparator or allow_smoke:
            raise ValueError('This loader is only for complete no-quality box comparator training')
        self.name = name
        self.folder = Path(output) / name
        self.sam = load_sam(name)
        head = GainSelector().cuda().eval().requires_grad_(False)
        state = torch.load(self.folder / 'box_selector.pth', map_location='cpu', weights_only=True)
        if (state.get('model'), state.get('component'), state.get('format_version')) != (name, 'box_selector', 2):
            raise ValueError('Box selector model/component mismatch')
        head.load_state_dict(state['state_dict'], strict=True)
        self.light = SimpleNamespace(sam=self.sam, engine=FourCandidateEngine(self.sam, name), box=head)
        self.decoder = copy.deepcopy(self.sam.mask_decoder).cuda().eval().requires_grad_(False)
        state = torch.load(self.folder / 'independent_decoder.pth', map_location='cpu', weights_only=True)
        if state.get('model') != name or state.get('format') != FORMAT:
            raise ValueError('Independent box decoder mismatch')
        self.decoder.load_state_dict(state['state_dict'], strict=True)
        self.comparator = None
        self.quality_heads = None
        self.last_trace = None
