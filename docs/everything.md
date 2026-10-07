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

The GitHub Pages explorer displays a recorded paired ViT-H comparison. Press
**Show recorded comparison** to reveal Dense SAM on the left and FSD-SAM on
the right, with measured milliseconds and completed-request counts. Both use
the same frozen ViT-H weights, image, 16×16 grid and filters. The standalone
process animation shows encoding, prompt factors, previews, suffix completion
and filtering, with replay, pause and manual step controls. Its dots and
playback duration are schematic, separate from measured inference time.

In the hosted/local studio, **Run both models** runs both paths on the current
image and shows their masks and times together. The generator dropdown is
removed; the Point/Box backbone chooser is hidden in Everything mode.
Each synchronized total is the shared measured image-encoding cost plus that
method's mask-generation cost. Loading, warm-up, queue and rendering are
excluded. Both methods are warmed before timing; repeat runs alternate order.
The **How FSD works** animation receives actual FSD request/mask counts.

Example times describe those individual runs and their execution device.
They are not dataset-average timings, speedup claims, or accuracy results.
Native SAM predicted IoU is a model score; uploaded images have no measured
ground-truth IoU. Existing point/box evaluation tables retain their protocol.
