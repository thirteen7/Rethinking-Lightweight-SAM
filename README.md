# Rethinking Lightweight SAM

**Prompt-Adaptive Refinement and Efficient Segment Everything Inference**

[Project page](https://thirteen7.github.io/Rethinking-Lightweight-SAM/) · [Interactive demo](https://huggingface.co/spaces/thirteen7/Rethinking-Lightweight-SAM) · [Models](https://github.com/thirteen7/Rethinking-Lightweight-SAM/releases/tag/v1.0.0) · [Results](docs/results.md) · [中文](README.zh-CN.md)

Code and models for **Rethinking Lightweight SAM with Prompt-Adaptive Refinement and Efficient Segment Everything Inference**.

The refinement interface supports **TinySAM, MobileSAM, and SAM ViT-H**. Each complete checkpoint includes the backbone and six refinement modules: first-click selection, interaction feedback, local residual repair, box selection, an independent box decoder, and an edit comparator.

## Installation

Python 3.10 or later. Install a compatible PyTorch/torchvision build first. Training and dataset evaluation use CUDA; model loading and the interactive API also support CPU.

```bash
git clone https://github.com/thirteen7/Rethinking-Lightweight-SAM.git
cd Rethinking-Lightweight-SAM
pip install -r requirements.txt
pip install -e . --no-deps
```

For experimental reproduction, use PyTorch 2.0.0 / torchvision 0.15.1 and keep the hardware, CUDA, and dependency versions fixed.

## Download models

```bash
python download_models.py --model tinysam
# Other options: mobilesam, vith, all
```

Checkpoints are saved to `weights/`. The downloader verifies SHA256 and automatically assembles the ViT-H transport parts into one complete `.pth`. Model hashes are listed in [models.json](models.json).

## Evaluate

```bash
python evaluate.py --model tinysam --checkpoint weights/tinysam_prompt_adaptive_v1.pth --dataset coco --data-root datasets/coco --output runs/tinysam-coco
```

Select `lvis` or `sa1b` with `--dataset`, and choose a GPU with `--gpu 1`. Use `--dry-run` to inspect data coverage or `--max-images 1` for an explicitly limited check.

Evaluation uses ordinary **legacy mIoU**, a mask-center first click, three point/box interaction rounds, and chunks of 64 objects. COCO/LVIS include all valid validation images and non-crowd targets. SA-1B uses the official cap64 reader: sampling without replacement, seeded by the original image index. Ground truth generates initial prompts, simulated corrective clicks, and IoU scores.

Outputs include `summary.json`, per-target numerical records, logs, and `verified.json`. Dataset evaluation does not save prediction images or masks.

```text
datasets/coco/
  annotations/instances_val2017.json
  annotations/lvis_v1_val.json
  val2017/
  train2017/                # Some LVIS validation images come from this split
datasets/sa1b/
  images/val/*.jpg
  annotations/val/*.json
```

A merged `coco/trainval/` image directory is also supported. LVIS evaluation uses `lvis_v1_val.json`, which includes images from both COCO train2017 and val2017.

## Train and export

Fitting, calibration, and checkpoint selection use **SA-1B train only**. The fixed 3,500-image manifest defines disjoint 2,900/300/300 partitions. First-click fitting uses 1,500 images; later decoder/comparator stages use 256/64/32 images. External validation labels are excluded from training.

```bash
python prepare_data.py --source datasets/sa1b --output datasets/sa1b_train3500
python download_models.py --model tinysam --base-only
python train.py --model tinysam --base weights/tinysam_official_base.pth --data-root datasets/sa1b_train3500 --output runs/train-tinysam
```

Training exports `runs/train-tinysam/tinysam_prompt_adaptive_v1.pth` automatically. Replace the model name to train another backbone. The default training decode chunk is 4; `--batch-size` changes the memory requirement.

Use `--reuse-first weights/tinysam_prompt_adaptive_v1.pth` to reuse a first-click module. By default, it is fitted from scratch. `--dry-run` inspects the inputs and prints the training stages.

To export an existing complete set of modules:

```bash
python merge.py --model tinysam --base weights/tinysam_official_base.pth --components runs/train-tinysam --output weights/tinysam_custom.pth
```

## Interactive API

```python
from PIL import Image
import numpy as np
from prompt_adaptive_sam import Predictor

predictor = Predictor("weights/tinysam_prompt_adaptive_v1.pth", device="cuda")
predictor.set_image(np.asarray(Image.open("example.jpg").convert("RGB")))
mask, logits, choice = predictor.predict([[200, 150]], [1])

# Pass the complete click history and the previous logits for correction.
mask, logits, choice = predictor.predict(
    [[200, 150], [250, 180]], [1, 0], previous=logits
)

# Boxes use original-image coordinates: [x0, y0, x1, y1].
mask, logits, choice = predictor.predict(box=[100, 80, 350, 300])
```

## Everything mode

Generate instance masks without supplying points or boxes. **FSD-SAM** uses
factorized prompt state, native previews and selective suffix decoding;
**Dense SAM** completes all sampled requests through the native decoder.

```bash
python generate.py --checkpoint weights/tinysam_prompt_adaptive_v1.pth --image example.jpg --method fsd --grid 32 --output runs/everything
```

The application supports TinySAM and MobileSAM, grid densities of 8/16/32,
colored instance overlays, and individual-mask inspection. See the
[Everything interface and protocol](docs/everything.md).

## Interactive application

The [project page](https://thirteen7.github.io/Rethinking-Lightweight-SAM/) includes a qualitative example explorer, interactive benchmark charts, and model downloads. The explorer displays recorded Point, Box, and Everything predictions.

For inference on uploaded images, run the same frontend with the Python backend:

```bash
pip install -r demo/requirements.txt
python demo/server.py --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`. The app supports foreground/background clicks, box prompts, previous-mask feedback, automatic Everything generation, and model switching. See [demo/README.md](demo/README.md) for the Gradio research studio and Hugging Face deployment.

## Results

Third-round ordinary mIoU (%). All three rounds and evaluation details appear in [docs/results.md](docs/results.md).

| Dataset | TinySAM point | TinySAM box | MobileSAM point | MobileSAM box |
|---|---:|---:|---:|---:|
| SA-1B official cap64 | 78.729 | 85.183 | 79.202 | 84.317 |
| COCO val2017, all targets | 69.438 | 78.364 | 69.502 | 77.228 |
| LVIS v1 val, all targets | 66.323 | 77.133 | 65.862 | 75.681 |

## License and acknowledgments

[LICENSE](LICENSE) · [NOTICE](NOTICE)

[SAM](https://github.com/facebookresearch/segment-anything) · [TinySAM](https://github.com/xinghaochen/TinySAM) · [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
