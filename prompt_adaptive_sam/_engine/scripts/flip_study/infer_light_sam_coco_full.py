# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
import torch
from torch.nn import functional as F
from qa_sam.utils.common import get_centroid_from_mask
from scripts.flip_study.eval_box_augmented_feedback import input_boxes

def prepare_chunk(dataset, image_index, start, end):
    (image, ann) = dataset[image_index, start, end]
    hw = tuple(map(int, ann['img_size_before_pad'][1:]))
    native_hw = (ann['info']['height'], ann['info']['width'])
    gt = ann['gt_mask'].float().cuda()[:, None]
    legacy = (F.interpolate(gt[:, :, :hw[0], :hw[1]], native_hw, mode='bilinear', align_corners=False) > 0)[:, 0]
    low_gt = F.interpolate(gt, (256, 256), mode='bilinear', align_corners=False)
    valid = torch.zeros(1, 1, 1024, 1024, device='cuda')
    valid[:, :, :hw[0], :hw[1]] = 1
    valid = F.interpolate(valid, (256, 256), mode='bilinear', align_corners=False)
    point = get_centroid_from_mask(gt > 0.5).float()
    labels = torch.ones(len(gt), 1, device='cuda')
    boxes = input_boxes(dataset.data[image_index][1][start:end], native_hw, hw, 'cuda')
    return (image[None].cuda(), hw, native_hw, low_gt, valid, legacy, point, labels, boxes)
