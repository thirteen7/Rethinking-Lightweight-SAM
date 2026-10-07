# Extracted from the verified research implementation; see LICENSE and NOTICE.
from qa_sam.utils.common import sample_point_in_mask

def original_next_point(low, gt_low, valid):
    """Same threshold, valid area, sampler and coordinate conversion as eval_mIoU.py."""
    assert low.shape[-2:] == gt_low.shape[-2:] == valid.shape[-2:] == (256, 256)
    pred_valid = (low > 0) * valid
    gt_valid = (gt_low > 0) * valid
    (xy, label) = sample_point_in_mask(pred_valid, gt_valid, num_samples=1, slic_labels=None)
    xy[..., 0] = xy[..., 0] / low.shape[3] * 1024
    xy[..., 1] = xy[..., 1] / low.shape[2] * 1024
    return (xy, label)
