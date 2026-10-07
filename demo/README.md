# Research studio

The same Gradio studio is used locally and in the Hugging Face Space.

## Run locally

Install the project and `demo/requirements.txt`, then run:

```bash
SAM_DEMO_DEVICE=cpu python demo/space/app.py
```

Windows PowerShell:

```powershell
$env:SAM_DEMO_DEVICE='cuda'  # use cpu without CUDA
python demo/space/app.py
```

The launcher detects the local checkout. Open `http://127.0.0.1:7860`.
Exact SHA256-verified complete checkpoints are downloaded only if missing.
The default CUDA path uses the local device for a local checkout, and ZeroGPU
in a hosted Space.

## Compare lightweight models

Point and Box use only the curated annotated black-bear, elephant, bottle and
bed examples. Select the mode/backbone or a gallery thumbnail, then press
**Run both models** once. Four panels show original/refined outputs at the
initial prompt and after two positive foreground corrections. Every stage's
true target IoU is recomputed from the supplied legacy GT mask. The same
prompts are sent to both paths with independent previous logits. There is no
background-point option and no custom-image upload in these two modes.

These are curated examples and prompt coordinates, not an estimate of average
performance. See [the comparison protocol](../docs/showcase.md).

## Compare automatic generation

Everything accepts image uploads or built-in examples. One button runs
SAM ViT-H / Dense on the left and FSD-SAM / ViT-H on the right, with the same
grid, frozen base and filters. CUDA-synchronized totals include shared image
encoding plus each generation time; loading, warm-up, queue and rendering are
excluded. Repeat runs alternate execution order. The current-image numbers
are separate from the paper's COCO100 dataset averages. Instance inspection
reports SAM's predicted IoU, not measured target IoU for an unlabeled upload.
The FSD animation has playback and step controls.

## Hugging Face deployment

Upload `app.py`, `README.md` and `requirements.txt` from `demo/space/` to the
existing Gradio Space after the corresponding project code is published.
The hosted launcher clones the public project into its versioned cache.
Keep ZeroGPU hardware; startup loads tensors and GPU prediction is scoped
to the comparison callback. No paid hardware is requested.

Gradio uses temporary storage for uploads and rendered outputs, cleaned on
a 15-minute schedule. Image state and measured masks stay in CPU session memory;
each paired run uses independent logits internally.
The older `demo/server.py` remains an API development utility; the Gradio
studio above is the public demonstration entry point.
