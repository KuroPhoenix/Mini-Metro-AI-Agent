"""
vision/preprocess.py
Utilities to convert Mini Metro observations into a fixed tensor.

Two modes:
    • 'pixels'   : raw RGB crop of the game canvas → (3, 96, 96) float32 [0,1]
    • 'symbolic' : env.world dict → 1-D float32 vector
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Literal
import numpy as np
from PIL import Image
import pyautogui          # used only in grab_screen()
import json
from config import game_bbox
# ---------- CONFIG ----------------------------------------------------------

# Bounding box is now read at runtime from config.yaml
TARGET_H, TARGET_W = 96, 96     # resize target for CNN

# pick the mode at runtime: preprocess(obs, mode="pixels")
Mode = Literal["pixels", "symbolic"]

# ---------- PIXEL PIPELINE --------------------------------------------------

def _grab_screen() -> Image.Image:
    """Returns a PIL Image of the GAME_BBOX area."""
    GAME_BBOX = game_bbox()
    im = pyautogui.screenshot(region=GAME_BBOX)
    return im.convert("RGB")

def _pixels_from_screenshot(img: Image.Image) -> np.ndarray:
    """
    Resize → CHW float32 in [0,1].
    """
    img = img.resize((TARGET_W, TARGET_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0       # HWC
    return np.transpose(arr, (2, 0, 1))                   # CHW  (3,96,96)

# ---------- SYMBOLIC PIPELINE ----------------------------------------------

FEATURE_ORDER = (
    "num_passengers",
    "num_trains",
    "num_lines",
    "num_tunnels",
    # … add whatever scalar features you expose in env.world
)

def _vector_from_world(world: Dict[str, Any]) -> np.ndarray:
    """
    Convert env.world (a dict) into a fixed-length numeric vector.
    Non-present keys default to 0.
    """
    vec = [float(world.get(k, 0.0)) for k in FEATURE_ORDER]

    # Encode each station as (type_id, x_norm, y_norm)
    # with at most 15 stations; pad with zeros.
    max_stations = 15
    stations = world.get("stations", [])
    for s in stations[:max_stations]:
        # s = {"type":"circle", "pos":(x,y)}
        type_id = {"circle": 0, "triangle": 1, "square": 2}.get(s["type"], 3)
        x, y = s["pos"]
        vec.extend([type_id, x / world["width"], y / world["height"]])
    # pad
    while len(stations) < max_stations:
        vec.extend([0.0, 0.0, 0.0])
        stations.append(None)

    return np.asarray(vec, dtype=np.float32)              # shape = (N,)

# ---------- PUBLIC ENTRY POINT ---------------------------------------------

def preprocess(obs: Any, mode: Mode = "pixels") -> np.ndarray:
    """
    `obs` is either:
        • None            – we grab a fresh screenshot (pixels mode only)
        • PIL.Image       – already a screenshot
        • env.world dict  – symbolic mode
    Returns a numpy array suitable for the agent.
    """
    if mode == "pixels":
        if obs is None or isinstance(obs, np.ndarray):   # allow raw states
            img = _grab_screen()
        elif isinstance(obs, Image.Image):
            img = obs
        else:
            raise TypeError("pixels mode expects PIL.Image or None")
        return _pixels_from_screenshot(img)

    elif mode == "symbolic":
        if not isinstance(obs, dict):
            raise TypeError("symbolic mode expects env.world dict")
        return _vector_from_world(obs)

    else:
        raise ValueError(f"unknown mode {mode}")
