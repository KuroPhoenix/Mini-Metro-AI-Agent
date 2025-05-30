
from __future__ import annotations
import pyautogui
import cv2
import numpy as np
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import networkx as nx
from vision.station_detector import annotate_stations
from vision.line_detector import annotate_lines
from .context_classification_tools.ContextClassifier.context_classifier import classify_context
from .context_classification_tools.ContextClassifier.ocr               import passenger_cnt
from actions.build_action_space import build_action_space
from actions.macro_definition import PAction
from config import game_bbox               # ← NEW
from PIL import Image

def screenshot_game_area(
    bbox: tuple[int, int, int, int] | None = None,
):
    """
    Grab the Steam window region once, return (PIL.Image, x0, y0).
    The bounding box can be supplied explicitly or falls back to config.yaml.
    """
    if bbox is None:
        bbox = game_bbox()
    left, top, width, height = bbox
    screenshot = pyautogui.screenshot(region=bbox)
    bgr_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return bgr_img, left, top


PIXEL_TOL = 4  # “how close” counts as touching


def _poly_touches_box(poly, box, tol=PIXEL_TOL) -> bool:
    """
    Quick-and-dirty spatial test:
    `poly` : list[(x, y)] ––  vertices of the line segment
    `box` : (x1, y1, x2, y2) –– station bbox
    `tol` : allowable gap in px (helps when detections almost touch)

    Returns True if *any* vertex falls inside the slightly enlarged box.
    """
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1 - tol, y1 - tol, x2 + tol, y2 + tol
    for x, y in poly:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────
# 1. Build a topological MultiGraph enriched with shapes & colours
# ─────────────────────────────────────────────────────────────────────
def connect_stations(
        boxes: list[tuple[int, int, int, int]],
        polys: list[list[tuple[int, int]]],
        colours: list[str],
        shapes: list[str],
) -> nx.MultiGraph:
    """
    Parameters
    ----------
    boxes   : station bounding boxes                len = Nₛ
    polys   : line-segment polygons                 len = Nₗ
    colours : line colour for each polygon          len = Nₗ
    shapes  : station shape label for each station  len = Nₛ

    Returns
    -------
    G : nx.MultiGraph
        • Node attributes : bbox, centre, shape
        • Edge attributes : colour  (one edge *per* segment)
    """
    G = nx.MultiGraph()

    # 1️⃣ add every station node
    for idx, (box, shape) in enumerate(zip(boxes, shapes)):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        G.add_node(
            idx,
            bbox=box,
            centre=(cx, cy),
            shape=shape,
        )

    # 2️⃣ group polygons by colour (helps readability; not strictly required)
    same_colour = defaultdict(list)
    for poly, col in zip(polys, colours):
        same_colour[col].append(poly)

    # 3️⃣ for every segment, connect all stations it touches
    for colour, segments in same_colour.items():
        for poly in segments:
            touched = [
                idx
                for idx, box in enumerate(boxes)
                if _poly_touches_box(poly, box)
            ]
            if len(touched) >= 2:  # segment links ≥2 stations
                for i in range(len(touched)):
                    for j in range(i + 1, len(touched)):
                        u, v = touched[i], touched[j]
                        # MultiGraph ⇒ each call adds a *new* edge
                        G.add_edge(u, v, colour=colour)

    return G

def extract_river_mask(bgr_img: np.ndarray) -> np.ndarray:
    """Return a binary mask where river pixels are 1."""
    assert bgr_img is not None, "screenshot is None!"
    assert isinstance(bgr_img, np.ndarray), f"Expected np.ndarray, got {type(bgr_img)}"
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # if we were handed a PIL Image, convert to BGR ndarray
    if isinstance(bgr_img, Image.Image):
        bgr_img = cv2.cvtColor(np.array(bgr_img), cv2.COLOR_RGB2BGR)

    # --- 1) Colour filter -----------------------------------------------------
    # Hue 90-130°, low-to-mid saturation, mid value.
    lower = np.array([ 90,  15,  40])   # tune ±5 per theme
    upper = np.array([130, 120, 180])
    mask  = cv2.inRange(hsv, lower, upper)

    # --- 2) Clean-up ----------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # --- 3) Keep the largest blob (defensive) ---------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255

    return mask   # uint8 {0,255}

# ─────────────────────────────────────────────────────────────────────
# River-crossing utility (can be imported by other modules)
# ─────────────────────────────────────────────────────────────────────
def count_river_crossings(p1: tuple[int, int],
                          p2: tuple[int, int],
                          river_mask: np.ndarray,
                          samples: int | None = None) -> int:
    """
    Return how many *bridges* are required for the straight segment p1→p2.

    A “bridge” is counted every time the polyline leaves dry land and
    **enters** the river (land→water transition).  So a single traverse
    of the river costs 1; going over a peninsula and re-entering the river
    costs 2, etc.

    Parameters
    ----------
    p1, p2     : world-coords (x, y) of segment endpoints
    river_mask : binary uint8 mask, 255 = water, 0 = land
    samples    : #points to probe along the segment.  Default = length(px)

    Returns
    -------
    int   number of bridges required
    """
    x1, y1 = p1
    x2, y2 = p2
    # choose a reasonable sampling density
    if samples is None:
        samples = max(10, int(math.hypot(x2 - x1, y2 - y1)))

    prev_water = bool(river_mask[y1, x1])
    bridges = 0

    for i in range(1, samples + 1):
        t = i / samples
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        in_water = bool(river_mask[y, x])
        # land → water ⇒ need one extra bridge
        if not prev_water and in_water:
            bridges += 1
        prev_water = in_water

    return bridges

def perceive(assets: Dict[str, int] | None = None) -> Tuple[Dict[str, Any], Any]:

    """
    Capture a fresh screenshot → run detectors → build a symbolic observation.
    """
    screenshot, x0, y0 = screenshot_game_area()
    if screenshot is None:
        raise ValueError("perceive() returned None for screenshot")
    # ↓ helper now returns: (annot_img, bboxes, shapes)
    annot_stations, boxes, station_shapes = annotate_stations(screenshot)

    # ↓ helper now returns: (annot_img, polygons, colours)
    annot_lines, polygons, line_colours = annotate_lines(screenshot)

    ctx, _, _ = classify_context(screenshot)

    graph = connect_stations(
        boxes=boxes,
        polys=polygons,
        colours=line_colours,
        shapes=station_shapes,
    )
    # 2. full node-attribute dump for quick lookup
    stations = {
        n: {
            "bbox": data["bbox"],
            "centre": data["centre"],
            "shape": data["shape"],
            "degree": graph.degree[n],
        }
        for n, data in graph.nodes(data=True)
    }
    # For every detected segment, remember which stations it touches
    segments: list[dict] = []
    for poly, colour in zip(polygons, line_colours):
        seg_stations = [
            idx
            for idx, box in enumerate(boxes)
            if _poly_touches_box(poly, box)
        ]
        # For every detected segment, remember which stations it touches
        # ➜ and the *exact coordinates* where the poly first meets each bbox
        segments: list[dict] = []
        for poly, colour in zip(polygons, line_colours):
            touched: list[int] = []
            endpoints: list[tuple[int, int]] = []
            # walk through the poly’s vertices in drawing order
            for vx, vy in poly:
                for idx, box in enumerate(boxes):
                    if idx in touched:  # station already logged
                        continue
                    if _poly_touches_box([(vx, vy)], box):
                        touched.append(idx)
                        endpoints.append((vx, vy))  # record contact point
                        break  # go test next vertex
                if len(touched) == 2:  # got both ends
                    break

            segments.append(
                {
                    "poly": poly,  # list[(x, y)]
                    "colour": colour,  # line colour label
                    "stations": touched,  # [u, v] (may be <2 if noisy)
                    "endpoints": endpoints  # [(x1,y1), (x2,y2)]  ← NEW
                }
            )

    unconnected_stations = [n for n, deg in graph.degree() if deg == 0]
    passengers = passenger_cnt(screenshot)
    river_mask = extract_river_mask(screenshot)
    actions: list[PAction] = []
    world = {
        "stations": stations or {},   # Never None
        "graph": graph,
        "unconnected": unconnected_stations,
        "segments": segments,
        "context": ctx,
        "passenger": passengers,
        "assets": assets,
        "river": river_mask,
        "actions": actions
    }
    return screenshot, world