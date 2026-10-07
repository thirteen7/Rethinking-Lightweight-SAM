# Extracted from the verified research implementation; see LICENSE and NOTICE.
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
from scripts.flip_study.train_local_attention import read
from scripts.flip_study.run_expanded_validation import COCODataset
DATA = ROOT / 'qasam-datasets/sa1b_selector_1500'

class TrainingDataset(COCODataset):

    def prepare_data(self):
        manifest = read(Path(self.data_root) / 'manifest.json')
        (self.data, self.keys, self.ignore_list) = ([], [], [])
        for row in manifest['images']:
            raw = read(Path(self.data_root) / 'annotations' / row['annotation_file'])
            image = dict(raw['image'])
            image['id'] = image.pop('image_id')
            assert image['id'] == row['image_id']
            annotations = [dict(a, image_id=image['id'], category_id=1, iscrowd=0) for a in raw['annotations']]
            assert len(annotations) == row['annotations'] and annotations
            self.data.append((image, annotations))
            self.keys.append(row['file_name'])
