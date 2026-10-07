"""Ephemeral image/prompt inference with the project's complete checkpoints."""
from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import io
import json
import logging
from pathlib import Path
import secrets
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
from prompt_adaptive_sam import Predictor
from download_models import download, sha256

LOG = logging.getLogger('prompt-adaptive-demo')
ALLOWED_MODELS = ('tinysam', 'mobilesam')
MAX_SESSIONS = 4
SESSION_TTL = 20 * 60


class SessionInput(BaseModel):
    image: str
    model: str = 'tinysam'


class SessionRef(BaseModel):
    session_id: str


class PredictionInput(SessionRef):
    points: list[dict] = Field(default_factory=list)
    box: list[float] | None = None


@dataclass
class Session:
    predictor: Predictor
    used: float = field(default_factory=time.monotonic)
    rounds: int = 0
    points: list = field(default_factory=list)
    box: list | None = None
    previous: np.ndarray | None = None


def create_app(device='cpu', weights=None):
    """No model allocation occurs until a visitor uploads an image."""
    weights = Path(weights) if weights else ROOT / 'weights'
    lock = threading.RLock()
    models = {}
    sessions = OrderedDict()
    index = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
    app = FastAPI(title='Rethinking Lightweight SAM', docs_url=None, redoc_url=None)
    torch.set_num_threads(2)

    def get_session(session_id):
        now = time.monotonic()
        for key in list(sessions):
            if now - sessions[key].used > SESSION_TTL:
                del sessions[key]
        if session_id not in sessions:
            raise HTTPException(404, 'Session expired. Upload the image again.')
        session = sessions[session_id]
        session.used = now
        sessions.move_to_end(session_id)
        return session

    @app.get('/', response_class=HTMLResponse)
    def home():
        html = (ROOT / 'site/index.html').read_text(encoding='utf-8')
        return html.replace('<body>', '<body data-live="true">', 1)

    @app.get('/api/health')
    def health():
        return dict(live=True, device=device, models=list(ALLOWED_MODELS),
                    persistence='temporary-memory', max_rounds=3)

    @app.post('/api/session')
    def session_create(request: SessionInput):
        if request.model not in ALLOWED_MODELS:
            raise HTTPException(400, 'Choose TinySAM or MobileSAM.')
        if len(request.image) > 8 * 1024 * 1024:
            raise HTTPException(413, 'Image is too large; use an image below 1800 pixels per side.')
        try:
            prefix, encoded = request.image.split(',', 1)
            if prefix not in ('data:image/jpeg;base64', 'data:image/png;base64', 'data:image/webp;base64'):
                raise ValueError('format')
            pixels = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(pixels)) as image:
                if image.width * image.height > 1800 * 1800 or max(image.size) > 1800 or min(image.size) < 8:
                    raise HTTPException(400, 'Use an image between 8 and 1800 pixels per side.')
                image_rgb = np.asarray(image.convert('RGB'))
        except HTTPException:
            raise
        except (ValueError, UnidentifiedImageError, OSError):
            raise HTTPException(400, 'Upload a valid JPEG, PNG, or WebP image.') from None
        with lock:
            if request.model not in models:
                record = index['models'][request.model]
                checkpoint = weights / record['filename']
                weights.mkdir(parents=True, exist_ok=True)
                url = f"https://github.com/{index['repository']}/releases/download/{index['release']}/{record['filename']}"
                download(url, checkpoint, record['sha256'], record['bytes'])
                if sha256(checkpoint) != record['sha256']:
                    raise RuntimeError('Model SHA256 differs from the release manifest')
                models[request.model] = Predictor(checkpoint, device=device)
                LOG.info('Loaded %s on %s', request.model, device)
            predictor = copy.copy(models[request.model])
            predictor.set_image(image_rgb)
            session_id = secrets.token_urlsafe(32)
            sessions[session_id] = Session(predictor=predictor)
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
            return dict(session_id=session_id, width=image_rgb.shape[1], height=image_rgb.shape[0], device=device)

    @app.post('/api/reset')
    def reset(request: SessionRef):
        with lock:
            session = get_session(request.session_id)
            session.points = []
            session.box = None
            session.rounds = 0
            session.previous = None
            return dict(reset=True)

    @app.post('/api/predict')
    def predict(request: PredictionInput):
        with lock:
            session = get_session(request.session_id)
            height, width = session.predictor.native_hw
            if session.rounds >= 3:
                raise HTTPException(400, 'Three rounds are complete. Reset prompts for another object.')
            if len(request.points) > 3:
                raise HTTPException(400, 'At most three point prompts are supported.')
            points = []
            for point in request.points:
                try:
                    x, y, label = float(point['x']), float(point['y']), point['label']
                except (KeyError, TypeError, ValueError):
                    raise HTTPException(400, 'Each point needs x, y, and a foreground/background label.') from None
                if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height) or label not in (0, 1):
                    raise HTTPException(400, 'Point coordinates or labels are invalid.')
                points.append([x, y, int(label)])
            box = request.box
            if box is not None:
                if len(box) != 4 or not np.isfinite(box).all() or not (0 <= box[0] < box[2] < width and 0 <= box[1] < box[3] < height):
                    raise HTTPException(400, 'Define a valid box inside the image.')
            expected = session.rounds + (0 if box is not None else 1)
            if len(points) != expected:
                raise HTTPException(400, 'Add exactly one corrective click for the next round.')
            if session.rounds:
                if points[:-1] != session.points or box != session.box:
                    raise HTTPException(400, 'Keep the prompt history and box fixed; reset to change the object.')
            elif box is None and (not points or points[0][2] != 1):
                raise HTTPException(400, 'Begin with a foreground point or a box.')
            started = time.perf_counter()
            mask, logits, choice = session.predictor.predict(
                [row[:2] for row in points], [row[2] for row in points], box=box,
                previous=session.previous)
            if not np.isfinite(logits).all() or mask.shape != (height, width):
                raise RuntimeError('Model output failed the shape/finite check')
            session.previous = logits
            session.points = points
            session.box = box
            session.rounds += 1
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[..., :3] = [118, 91, 234]
            rgba[..., 3] = mask.astype(np.uint8) * 255
            stream = io.BytesIO()
            Image.fromarray(rgba, 'RGBA').save(stream, format='PNG', optimize=True)
            return dict(mask='data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode(),
                        candidate=choice, round=session.rounds,
                        seconds=time.perf_counter() - started)

    app.mount('/', StaticFiles(directory=ROOT / 'site'), name='site')
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    import uvicorn
    uvicorn.run(create_app(args.device, args.weights), host=args.host, port=args.port)
