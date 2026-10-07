---
title: Rethinking Lightweight SAM
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: '3.10'
app_file: app.py
pinned: false
license: other
---

# Rethinking Lightweight SAM with Prompt-Adaptive Refinement and Efficient Segment Everything Inference

[Project page](https://thirteen7.github.io/Rethinking-Lightweight-SAM/) ·
[Code and models](https://github.com/thirteen7/Rethinking-Lightweight-SAM)

Press **Run both models** in **Everything** to generate a left-hand **SAM ViT-H / Dense SAM**
result and a right-hand **FSD-SAM / ViT-H** result, with measured milliseconds for each.
Both paths use the same frozen ViT-H weights, input image, sampling grid, FP32 precision,
and native quality, stability and NMS thresholds. There is no generator dropdown.
The totals include the same measured image encoding time plus each mask-generation time.
CUDA is synchronized around timing. Warm-up, model loading, the ZeroGPU queue and
overlay rendering are excluded. Repeated runs alternate execution order.
These are demo measurements on the current device, not the paper's fused-kernel benchmark.

Choose TinySAM or MobileSAM for **Point** or **Box**. These modes use only four
curated annotated examples: black bear, elephant, bottle and bed. One Run
executes both models with the same fixed foreground prompts and independent
mask feedback, showing initial decoder outputs, outputs after two positive
corrections, and true target IoU at every stage. The background-point selector
has been removed. No custom-image uploads are accepted in Point or Box.
Curated examples illustrate successful cases; full dataset averages are listed
separately on the project page. [Comparison protocol](https://github.com/thirteen7/Rethinking-Lightweight-SAM/blob/main/docs/showcase.md).

**Everything** keeps image uploads, opacity controls and an FSD instance inventory.
The **How FSD works** tab animates both paths with playback controls. Animation
duration is separate from measured inference time. Inference runs on ZeroGPU;
uploads and predictions use a temporary cache.

Everything uses the portable PyTorch decoder and fixed native SAM filters.
Colors represent instances, not semantic classes. Uploaded images have no
measured ground-truth IoU score.

License: see the project's LICENSE and NOTICE.
