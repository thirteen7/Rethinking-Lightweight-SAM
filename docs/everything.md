# Everything mode

Everything generates instance masks without user clicks, boxes, or annotation
prompts. TinySAM and MobileSAM use their frozen SAM backbone and its three
public candidates. Instance colors distinguish masks, not semantic categories.

## Run

```bash
python download_models.py --model tinysam
python generate.py --checkpoint weights/tinysam_prompt_adaptive_v1.pth \
  --image example.jpg --method fsd --grid 32 --output runs/everything
```

Use `--device cuda` for a compatible CUDA installation. Choose `--method dense`
for native dense SAM decoding. The Python interface is:

```python
from prompt_adaptive_sam import Predictor, EverythingGenerator

p = Predictor('weights/tinysam_prompt_adaptive_v1.pth', device='cuda')
masks, details = EverythingGenerator(p, method='fsd', points_per_side=32).generate(rgb)
```

Each output includes a COCO-format RLE mask, pixel area, bounding box, the native
SAM predicted IoU, and stability score. No extra scoring head is added.

## Inference paths

**FSD-SAM** computes complete independent prompt factors, native phase-(0,0)
previews, four-wave selection, and local small-region protection. Retained
requests complete native suffix decoding. The hosted application uses the
portable PyTorch implementation; it does not claim the custom fused CUDA
kernel timings reported in experiments. Preview selection is approximate.

**Dense SAM** completes each sampled request through the native decoder.
Both paths use fixed predicted-IoU 0.88, stability 0.95, NMS 0.7, three public
mask candidates, and no image crops. The grid is 8×8, 16×16, or 32×32.
Interactive demos default to 16×16; the command-line default is 32×32.
These grid settings change the sampling density.

## Website and application

The GitHub Pages explorer displays recorded, prompt-free predictions. Switch
to Everything and select individual instances to inspect their geometry.
The hosted/local research studio additionally runs uploaded images, offers
both generators, and shows the actual instance inventory and inference time.

Example times describe those individual runs and their execution device.
They are not dataset-average timings, speedup claims, or accuracy results.
Native SAM predicted IoU is a model score; uploaded images have no measured
ground-truth IoU. Existing point/box evaluation tables retain their protocol.
