# Extracted from the verified research implementation; see LICENSE and NOTICE.
import torch

def cap_dataset(dataset):
    """Freeze the legacy index-seeded randint sampler, including replacement."""
    records = []
    for (index, (info, annotations)) in enumerate(dataset.data):
        selected = list(range(len(annotations)))
        if len(selected) > 64:
            generator = torch.Generator().manual_seed(index)
            selected = torch.randint(0, len(annotations), (64,), generator=generator).tolist()
        dataset.data[index] = (info, [annotations[j] for j in selected])
        records.append(dict(image_id=info['id'], available=len(annotations), selected=len(selected), unique=len(set(selected)), annotation_ids=[annotations[j]['id'] for j in selected]))
    return records
