# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
from scripts.flip_study.prepare_local_attention import cap_dataset
from scripts.flip_study.train_sa1b_selector import TrainingDataset
DATA = ROOT / 'qasam-datasets/sa1b_selector_3500'
SPLITS = {'fit': (0, 2900), 'dev': (2900, 3200), 'test': (3200, 3500)}

def dataset():
    data = TrainingDataset(str(DATA), split='train', load_gt_mask=True, max_allowed_prompts=-1, fix_seed=True)
    cap_dataset(data)
    if len(data) != 3500:
        raise ValueError(f'Expected 3500 disjoint SA-1B images, got {len(data)}')
    return data
