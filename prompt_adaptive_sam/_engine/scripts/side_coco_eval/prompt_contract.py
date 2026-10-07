# Extracted from the verified research implementation; see LICENSE and NOTICE.
from pathlib import Path
import hashlib
VERSION = 'minus_two_zero_v1'
ROOT = Path(__file__).resolve().parents[2]

def prompt_contract():
    names = ['qa_sam/modeling/prompt_encoder.py', '.local-deps/TinySAM/tinysam/modeling/prompt_encoder.py', '.local-deps/MobileSAM/mobile_sam/modeling/prompt_encoder.py', 'scripts/flip_study/light_sam_adapter.py', 'qa_sam/utils/common.py', 'scripts/flip_study/eval_legacy_box_versions.py', 'scripts/side_coco_eval/prompt_contract.py']
    return dict(version=VERSION, ignored_label=-2, ignored_embedding='zero; token retained', labels_unchanged=[-1, 0, 1], training_and_inference_shared=True, source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names})
