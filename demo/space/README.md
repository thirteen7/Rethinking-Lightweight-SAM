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

# Rethinking Lightweight SAM

Prompt-Adaptive Refinement and Efficient Segment Everything Inference.

[Project page](https://thirteen7.github.io/Rethinking-Lightweight-SAM/) ·
[Code and models](https://github.com/thirteen7/Rethinking-Lightweight-SAM)

Choose TinySAM or MobileSAM and explore **Point**, **Box**, or **Everything**.
The research studio includes built-in images, FSD-SAM and Dense SAM generation,
independent instance colors, opacity controls, and a mask inventory.
It opens with a recorded Orange bowl preview; run segmentation to generate
a new result. Songbird and Brown bear photographs are also included.
Inference runs on ZeroGPU. Uploads and predictions use temporary
Space storage; image state and previous logits remain in session memory.

Everything uses the portable PyTorch decoder and fixed native SAM filters.
Colors represent instances, not semantic classes. Uploaded images have no
measured ground-truth IoU score.

License: see the project's LICENSE and NOTICE.
