# Extracted from the verified research implementation; see LICENSE and NOTICE.
import torch

def input_boxes(records, native_hw, input_hw, device):
    xywh = torch.tensor([record['bbox'] for record in records], dtype=torch.float32, device=device)
    boxes = xywh.clone()
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, [0, 2]] *= input_hw[1] / native_hw[1]
    boxes[:, [1, 3]] *= input_hw[0] / native_hw[0]
    assert torch.isfinite(boxes).all() and (boxes[:, 2:] > boxes[:, :2]).all()
    return boxes
