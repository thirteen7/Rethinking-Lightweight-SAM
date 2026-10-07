# Interactive application

The project frontend can run as a static example explorer or as an image-upload
application backed by the complete TinySAM and MobileSAM checkpoints.

## Local inference

Install the project, then:

```bash
pip install -r demo/requirements.txt
python demo/server.py --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`. Upload an image, select a backbone, and add a
foreground click or drag a box. Run prediction, then add one corrective click
for each additional round. Reset prompts to select another object.

The application downloads a checkpoint only if its exact SHA256-verified file
is missing. Image embeddings, click histories, and logits remain in a bounded
in-memory session cache; uploaded images and predictions are not written to
disk. There is no IoU score for uploaded images without a ground-truth mask.

The default is CPU. Use `--device cuda` with a compatible CUDA installation.
Benchmark values on the project page remain fixed evaluation results and are
independent of interactive application timings.

## Hugging Face Space

Create a public **Gradio** Space under `thirteen7/Rethinking-Lightweight-SAM`.
Upload `app.py`, `README.md`, and `requirements.txt` from `demo/space/` into the
Space's repository root. The launcher obtains the public project code and
serves the same frontend with the inference API. Use the free available
hardware; the application performs CPU inference and does not request paid
hardware or a GPU allocation.

Set `site/config.json` → `space_url` to the Space's public address after the
application is running. The project page then opens the hosted interactive app.
