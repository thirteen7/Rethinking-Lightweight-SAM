# Extracted from the verified research implementation; see LICENSE and NOTICE.
import torch
from scripts.flip_study.box_specialist import box_features
from scripts.flip_study.candidate_selector import prediction_features
QUALITY_COLUMNS = (16, 20)
CANDIDATE_ID_COLUMNS = (23, 27)

def choose_anchor(prediction, compatible):
    """0 means actual current output; 1..4 mean new decoder tokens 0..3."""
    assert prediction.shape[1:] == (4, 3)
    assert compatible.shape == prediction.shape[:2]
    score = prediction[..., :2].min(-1).values
    eligible = compatible & torch.isfinite(prediction).all(-1) & (score > 0)
    score = score.masked_fill(~eligible, -torch.inf)
    return torch.cat([score.new_zeros(len(score), 1), score], 1).argmax(1)

def pair_features(current_low, current_quality, current_tokens, new_low, new_quality, new_tokens, boxes, clicks, labels, hw):
    """Pack two BoxSelectorV2 groups, preserving canonical new candidate IDs.

    Group 0 = [current selected, new0, new1, new2].
    Group 1 = [current selected, new3, new3, new3]; only its first output is used.
    prediction_features's baseline columns 8:16 contain the selected current
    candidate's statistics.  Columns 16:20 are overwritten with ALL FOUR new
    decoder qualities, and 23:27 with true new candidate eye4 IDs.  Thus padding
    and grouping cannot redefine a candidate's identity or quality context.
    """
    n = len(boxes)
    assert current_low.shape == (n, 1, 256, 256)
    assert current_quality.shape == (n, 1) and current_tokens.shape == (n, 1, 256)
    assert new_low.shape == (n, 4, 256, 256)
    assert new_quality.shape == (n, 4) and new_tokens.shape == (n, 4, 256)
    assert clicks.shape == (n, labels.shape[1], 2) and labels.shape[1] <= 2
    all_low = torch.cat([current_low, new_low], 1)
    all_quality = torch.cat([current_quality, new_quality], 1)
    all_tokens = torch.cat([current_tokens, new_tokens], 1)
    coords = torch.cat([boxes.reshape(n, 2, 2).mean(1)[:, None], clicks], 1)
    labs = torch.cat([labels.new_full((n, 1), -1.0), labels], 1)
    (token_groups, stat_groups, box_groups, compat_groups) = ([], [], [], [])
    eye = torch.eye(4, device=new_low.device, dtype=new_low.dtype)
    for (indices, candidate_ids) in [([0, 1, 2, 3], [0, 1, 2]), ([0, 4, 4, 4], [3, 3, 3])]:
        (low, quality) = (all_low[:, indices], all_quality[:, indices])
        (stats, compatible) = prediction_features(low, quality, coords, labs, hw, labs.shape[1])
        stats[..., QUALITY_COLUMNS[0]:QUALITY_COLUMNS[1]] = new_quality[:, None, :]
        stats[..., CANDIDATE_ID_COLUMNS[0]:CANDIDATE_ID_COLUMNS[1]] = eye[candidate_ids][None]
        token_groups.append(all_tokens[:, indices])
        stat_groups.append(stats)
        box_groups.append(box_features(low, boxes, hw))
        compat_groups.append(compatible)
    compatible = torch.cat([compat_groups[0], compat_groups[1][:, :1]], 1)
    finite = torch.isfinite(new_low).flatten(2).all(-1) & torch.isfinite(new_quality) & torch.isfinite(new_tokens).all(-1)
    return (torch.stack(token_groups, 1), torch.stack(stat_groups, 1), torch.stack(box_groups, 1), compatible & finite)
