import cv2
from typing import List, Tuple, Union

import os, inference
import supervision as sv
import numpy as np
os.environ["ROBOFLOW_API_KEY"] = "xtlx7woPbWGVbe2AR1qS"
model = inference.get_model("minimetrostation/1")


def annotate_stations(
    image: Union[str, np.ndarray],
    *,
    conf: float = 0.84,
    overlap: float = 0.6,
) -> Tuple[np.ndarray,                 # annotated image
           List[Tuple[int, int, int, int]],  # bounding boxes
           List[str]]:                       # shape / class labels
    """
    Detects stations in `image`, draws the bounding boxes & labels, and
    returns (1) the annotated image, (2) bounding-box coords and
    (3) the class labels.

    Parameters
    ----------
    image : str | numpy.ndarray
        • File path / URL (str) _or_
        • BGR image already loaded via cv2 / Pillow / etc. (numpy.ndarray)
    conf : float
        Confidence threshold for Roboflow inference.
    overlap : float
        NMS / overlap threshold supplied to the API.
    """
    # 1. Obtain a BGR image ndarray (`img`) irrespective of the input type
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image!r}")
    else:
        img = image.copy()                     # avoid mutating caller’s array

    # 2. Run Roboflow inference
    rf_result = model.infer(img, confidence=conf, overlap=overlap)[0]

    # 3. Convert to Supervision objects
    detections = sv.Detections.from_inference(rf_result)
    labels: List[str] = [pred.class_name for pred in rf_result.predictions]

    # 4. Extract (x1, y1, x2, y2) integer bounding boxes
    boxes: List[Tuple[int, int, int, int]] = [
        (int(x1), int(y1), int(x2), int(y2))
        for x1, y1, x2, y2 in detections.xyxy
    ]

    # 5. Annotate the image
    annotated_image = img.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image,
                                                   detections, labels)

    # Optional preview
    #sv.plot_image(annotated_image)

    return annotated_image, boxes, labels


