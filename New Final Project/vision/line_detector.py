"""
Line-segment detector and visualiser.

The function ``annotate_lines``

    • accepts either a file path / URL **or** an already-loaded BGR image
    • performs Roboflow instance-segmentation
    • draws masks, polygon outlines and labels on a *copy* of the image
    • returns
        (1) the annotated image,
        (2) a list of polygons            – list[list[(x, y)]]
        (3) a list of colour / class tags – list[str]

All heavy objects (Roboflow model, annotators) are created **once** and reused.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np
import supervision as sv           # drawing helpers
import inference                    # Roboflow client


# ──────────────────────────────────────────────────────────────────────────────
#  1.  Model initialisation (lazy – first call triggers the download)
# ──────────────────────────────────────────────────────────────────────────────
_ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "xtlx7woPbWGVbe2AR1qS")
os.environ["ROBOFLOW_API_KEY"] = _ROBOFLOW_API_KEY   # required by the SDK

_model: Optional["inference.Model"] = None


def _get_model() -> "inference.Model":
    """Return a cached Roboflow model instance (download on first use)."""
    global _model
    if _model is None:
        _model = inference.get_model("mini-metro-line-detect-v2/2")
    return _model


# ──────────────────────────────────────────────────────────────────────────────
#  2.  Single global annotators (instantiated once → reused)
# ──────────────────────────────────────────────────────────────────────────────
def _build_mask_annotator(transparency: float = 0.4) -> sv.MaskAnnotator:
    """
    Create a MaskAnnotator that works with multiple versions of the supervision
    library.  Newer versions accept `alpha`; some older ones accept `opacity`;
    the very old ones accept neither.  We probe the signature at runtime and
    fall back to the default constructor if necessary.
    """
    for kw in ("alpha", "opacity"):
        try:
            return sv.MaskAnnotator(**{kw: transparency})
        except TypeError:
            continue
    return sv.MaskAnnotator()


_MASK_ANN  = _build_mask_annotator(0.4)
_POLY_ANN  = sv.PolygonAnnotator(thickness=2)
_LABEL_ANN = sv.LabelAnnotator()


# ──────────────────────────────────────────────────────────────────────────────
#  3.  Public API
# ──────────────────────────────────────────────────────────────────────────────
def _xy_of_point(pt) -> Tuple[int, int]:
    """
    Robustly return integral (x, y) from *either*
    • a dict with "x"/"y" keys (older SDK)  *or*
    • a Point object with .x / .y attributes (current SDK).
    """
    if isinstance(pt, dict):                       # legacy
        return int(pt["x"]), int(pt["y"])
    # new SDK – attribute access
    return int(getattr(pt, "x")), int(getattr(pt, "y"))


def annotate_lines(
    image: Union[str, np.ndarray],
    *,
    conf: float = 0.50,
    overlap: float = 0.40,
    show: bool = False,
) -> Tuple[np.ndarray,                   # annotated BGR image
           List[List[Tuple[int, int]]],  # polygons
           List[str]]:                   # colour / class labels
    """
    Detect line *segments* in the supplied image and overlay their masks.

    Returns
    -------
    annotated_img : np.ndarray
    polygons      : list[list[(x, y)]]
    colours       : list[str]
    """
    # ---------------------------------------------------------------------- #
    # 0.  Load / duplicate the input image                                   #
    # ---------------------------------------------------------------------- #
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image!r}")
    else:
        img = image.copy()                             # do *not* mutate caller

    # ---------------------------------------------------------------------- #
    # 1.  Roboflow inference (instance-segmentation)                         #
    # ---------------------------------------------------------------------- #
    model = _get_model()
    result = model.infer(img, confidence=conf, overlap=overlap)

    if not result:                        # API always returns a list of frames
        h, w = img.shape[:2]
        return img, [], []

    rf_frame = result[0]                  # first (and only) frame

    # ---------------------------------------------------------------------- #
    # 2.  Build a Supervision Detections object                              #
    # ---------------------------------------------------------------------- #
    detections = sv.Detections.from_inference(rf_frame)

    # Roboflow → polygons + labels
    polygons: List[List[Tuple[int, int]]] = [
        [_xy_of_point(pt) for pt in getattr(pred, "points", [])]
        for pred in rf_frame.predictions
    ]
    colours: List[str] = [pred.class_name for pred in rf_frame.predictions]

    # ---------------------------------------------------------------------- #
    # 3.  Draw overlays                                                      #
    # ---------------------------------------------------------------------- #
    annotated = img.copy()
    annotated = _MASK_ANN.annotate(annotated, detections)
    annotated = _POLY_ANN.annotate(annotated, detections)
    annotated = _LABEL_ANN.annotate(annotated, detections, colours)

    if show:
        cv2.imshow("Detected lines", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return annotated, polygons, colours