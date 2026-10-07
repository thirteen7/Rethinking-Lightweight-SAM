# Extracted from the verified research implementation; see LICENSE and NOTICE.
from __future__ import annotations
import math
import time

def format_progress(label: str, done: int, total: int, started: float, detail: str='', processed_now: int | None=None) -> str:
    if processed_now is None:
        processed_now = done
    if total <= 0 or not 0 <= processed_now <= done <= total:
        raise ValueError(f'Invalid progress: {done}/{total}')
    elapsed = max(time.perf_counter() - started, 0.001)
    rate = processed_now / elapsed
    eta = f'{math.ceil((total - done) / rate)}s' if rate else '--'
    filled = done * 50 // total
    bar = '>' * filled + ' ' * (50 - filled)
    return f'{label} [{bar}] {done}/{total}, {rate:.2f} task/s, elapsed: {math.floor(elapsed)}s, ETA: {eta}{detail}'
