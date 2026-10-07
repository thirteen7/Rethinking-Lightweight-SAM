# Extracted from the verified research implementation; see LICENSE and NOTICE.
from scripts.flip_study.click_residual import PATCH, CHANNELS
from scripts.flip_study.train_sa1b_selector import DATA, TrainingDataset
from scripts.flip_study.train_local_attention import read
import torch
from torch.nn import functional as F
CFG = dict(images=256, objects_per_image=16, seed=11, sampling_seed=2026092608, stages=2, epochs_per_stage=3, batch_size=32, lr=0.001, weight_decay=0.0001, click_seeds=[11, 23], patch=PATCH, channels=CHANNELS, max_delta=8.0, dice_weight=0.5, harm_weight=0.5, preservation_weight=0.01)

class PilotDataset(TrainingDataset):

    def __init__(self, rows):
        self.rows = rows
        super().__init__(str(DATA), split='train', load_gt_mask=True, max_allowed_prompts=-1, fix_seed=True)

    def prepare_data(self):
        (self.data, self.keys, self.ignore_list) = ([], [], [])
        for row in self.rows:
            raw = read(DATA / 'annotations' / row['annotation_file'])
            info = dict(raw['image'])
            info['id'] = info.pop('image_id')
            annotations = [dict(raw['annotations'][j], image_id=info['id'], category_id=1, iscrowd=0) for j in row['selected_indices']]
            assert [a['id'] for a in annotations] == row['annotation_ids']
            self.data.append((info, annotations))
            self.keys.append(row['file_name'])

def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)

def loss_fn(model, batch):
    (inputs, base, support, target, area) = [batch[key].cuda().float() for key in ['inputs', 'base', 'support', 'target', 'area']]
    delta = model(inputs, support)
    refined = base + delta
    valid = (support > 0).float()
    count = valid.sum((1, 2, 3)).clamp_min(1)
    active = (valid.sum((1, 2, 3)) > 0).float()
    average = lambda x: (x * valid).sum((1, 2, 3)) / count
    bce = average(F.binary_cross_entropy_with_logits(refined, target, reduction='none'))
    base_bce = average(F.binary_cross_entropy_with_logits(base, target, reduction='none'))
    prob = refined.sigmoid() * valid
    truth = target * valid
    dice = 1 - (2 * (prob * truth).sum((1, 2, 3)) + 1) / (prob.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)
    harm = (bce - base_bce).clamp_min(0) * (1 + (area >= 1024).float())
    correct = ((base > 0) == (target > 0.5)).float()
    preserve = average(delta.abs() * correct)
    loss = bce + CFG['dice_weight'] * dice + CFG['harm_weight'] * harm + CFG['preservation_weight'] * preserve
    return ((loss * active).sum() / active.sum().clamp_min(1), {'bce': float((bce * active).sum().detach() / active.sum().clamp_min(1)), 'active': int(active.sum())})
