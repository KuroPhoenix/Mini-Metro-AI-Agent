# config.py  (put it in project root, next to mini_metro_env.py)

from pathlib import Path
import os, yaml, functools
from typing import Tuple

_DEFAULT_BBOX = (0, 29, 1280, 828)   # left, top, width, height

@functools.lru_cache(maxsize=1)
def load() -> dict:
    """
    Load YAML once and cache it.
    Fallback to sensible defaults when the file/keys are missing.
    """
    path = Path(os.getenv("MINIMETRO_CONFIG", "config.yaml"))
    if not path.exists():
        return {"bounding_box": _DEFAULT_BBOX}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("bounding_box", _DEFAULT_BBOX)
    return data

def game_bbox() -> Tuple[int, int, int, int]:
    """Return (left, top, width, height) as pyautogui expects."""
    return tuple(load()["bounding_box"])
