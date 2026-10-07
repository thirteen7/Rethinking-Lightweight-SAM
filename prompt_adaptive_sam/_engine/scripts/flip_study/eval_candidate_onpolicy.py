# Extracted from the verified research implementation; see LICENSE and NOTICE.
from qa_sam.utils.common import get_centroid_from_mask
from pycocotools import mask as mask_utils
import numpy as np
import torch
from torch.nn import functional as F

def chunk_inputs(dataset, index, start, end, device='cuda'):
    """Use dataset's annotation slicing; never sample/truncate dense LVIS images."""
    (image, annotation) = dataset[index, start, end]
    hw = tuple(map(int, annotation['img_size_before_pad'][1:]))
    info = annotation['info']
    native_hw = (info['height'], info['width'])
    records = dataset.data[index][1][start:end]
    gt = annotation['gt_mask'].float().to(device)[:, None]
    truth = (F.interpolate(gt[:, :, :hw[0], :hw[1]], native_hw, mode='bilinear', align_corners=False) > 0)[:, 0]
    masks = []
    for r in records:
        seg = r['segmentation']
        if isinstance(seg, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(seg, *native_hw))
        elif isinstance(seg['counts'], list):
            rle = mask_utils.frPyObjects(seg, *native_hw)
        else:
            rle = seg
        masks.append(mask_utils.decode(rle).astype(bool))
    native = torch.from_numpy(np.stack(masks)).to(device)
    valid = torch.zeros(1, 1, 1024, 1024, device=device)
    valid[:, :, :hw[0], :hw[1]] = 1
    valid = F.interpolate(valid, (256, 256), mode='bilinear', align_corners=False)
    low_gt = F.interpolate(gt, (256, 256), mode='bilinear', align_corners=False)
    coords = get_centroid_from_mask(gt > 0.5).float()
    labels = torch.ones(len(gt), 1, device=device)
    return (image[None].to(device), hw, native_hw, low_gt, valid, truth, native, [r['id'] for r in records], coords, labels)
