import cv2
from typing import List, Tuple, Union

import inference
import supervision as sv
import numpy as np

model = inference.get_model("minimetrostation/1")

def annotate_stations(
    image: Union[str, np.ndarray]
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Detects stations in `image`, draws the bounding boxes & labels, and
    returns (1) the annotated image and (2) the list of pixel coordinates
    for each detection.

    Parameters
    ----------
    image : str | numpy.ndarray
        • File path / URL (str) _or_
        • BGR image already loaded via cv2 / Pillow / etc. (numpy.ndarray)

    Returns
    -------
    annotated_img : numpy.ndarray
        Image with boxes + class labels drawn on it.
    boxes : list[tuple[int, int, int, int]]
        Bounding-box coordinates as (x_min, y_min, x_max, y_max) in pixels.
    """
    # --------------------------------------------------------------- #
    # 1. Read the image if a path/URL was given.                      #
    # --------------------------------------------------------------- #
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image!r}")
    else:
        img = image.copy()  # avoid mutating caller’s array

    # run inference on our chosen image
    predictions = model.infer(image, confidence=0.84)[0]
    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(predictions)
    labels = [prediction.class_name for prediction in predictions.predictions]

    # create supervision annotators
    boxes: List[Tuple[int, int, int, int]] = [
        (int(x1), int(y1), int(x2), int(y2))  # ← length is now obvious
        for x1, y1, x2, y2 in detections.xyxy
    ]

    # annotate the image with our inference results
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    # display the image
    sv.plot_image(annotated_image)
    return annotated_image, boxes
