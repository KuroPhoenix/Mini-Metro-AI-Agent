import cv2
from typing import List, Tuple, Union

import inference                       # Roboflow inference client
import supervision as sv               # annotation helpers
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Load the Roboflow line-segmentation model once at import time
# ──────────────────────────────────────────────────────────────────────────────
model = inference.get_model("mini-metro-line-detect-v2/1")


def annotate_lines(
    image: Union[str, np.ndarray],
    *,
    conf: float,
    overlap: float,
) -> Tuple[np.ndarray,                 # annotated BGR image
     List[List[Tuple[int, int]]],  # polygons
     List[str]]:                   # class / colour labels

    """
    Detects *lines* (instance-segmentation model) in `image`,
    overlays polygons + labels, and returns

        (1) the annotated image (BGR, uint8)
        (2) a list of polygons, one per detection, where each polygon is a
            list of (x, y) pixel tuples *in image coordinates*.

    Parameters
    ----------
    image : str | numpy.ndarray
        • Path / URL (str)                      ──› will be read with cv2.imread
        • OR an image already loaded in BGR     ──› will be copied, not mutated
    conf : float, default 0.50
        Confidence threshold for Roboflow inference.
    overlap : float, default 0.4
        NMS/overlap threshold supplied to the API.

    Returns
    -------
    annotated_img : numpy.ndarray
        Image with segmentation masks *and* class labels drawn.
    polygons : list[list[tuple[int, int]]]
        Polygon vertices for every instance, in the same order as rendered.
    """
    # ──────────────────────────────────────────────────────────────────────
    # 1. Load / duplicate the image
    # ──────────────────────────────────────────────────────────────────────
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image!r}")
    else:
        img = image.copy()                 # avoid mutating caller’s ndarray

    # ──────────────────────────────────────────────────────────────────────
    # 2. Run Roboflow inference (instance segmentation)
    #    `model.infer` returns a list; grab element 0 (the first frame)
    # ──────────────────────────────────────────────────────────────────────
    rf_result = model.infer(img,            # you *can* also pass the path/URL
                            confidence=conf,
                            overlap=overlap)[0]

    # ──────────────────────────────────────────────────────────────────────
    # 3. Convert to a Supervision `Detections` object (handles polygons)
    # ──────────────────────────────────────────────────────────────────────
    detections = sv.Detections.from_inference(rf_result)

    # Roboflow’s `.predictions` list still carries the raw objects so we
    # can pull polygons and labels straight from there:
    polygons: List[List[Tuple[int, int]]] = [
        [(int(pt["x"]), int(pt["y"])) for pt in pred.points]
        for pred in rf_result.predictions
    ]
    labels = [pred.class_name for pred in rf_result.predictions]

    # ──────────────────────────────────────────────────────────────────────
    # 4. Draw masks / polygons and labels
    #    • `MaskAnnotator` gives you a solid-fill overlay
    #    • `PolygonAnnotator` draws only the outline (pick the one you like)
    # ──────────────────────────────────────────────────────────────────────
    annotated = img.copy()

    # (a) fill the mask with semi-transparent colour
    annotated = sv.MaskAnnotator(alpha=0.4).annotate(annotated, detections)

    # (b) add the polygon outline for crisp edges (optional)
    annotated = sv.PolygonAnnotator(thickness=2).annotate(annotated, detections)

    # (c) finally, label each instance
    annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)

    # Preview (convenience; comment out if you don’t want an immediate pop-up)
    sv.plot_image(annotated)

    return annotated, polygons, labels

