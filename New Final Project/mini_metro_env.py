import pyautogui
from collections import defaultdict
import networkx as nx
from station_detector import annotate_stations
from line_detector import annotate_lines
from context_classification_tools.ContextClassifier.context_classifier import classify_context
from context_classification_tools.ContextClassifier.ocr import passenger_cnt
from actions.macro_definition import (Verb, PAction, Speed, Reward)

class MiniMetroEnv:
    def __init__(self):
        self.lines = {}
        self.world = {}
        self.perceive()
        self.connected_pairs = set()
        self.actions: list[PAction] = self._build_action_space()
        self.action_space_n = len(self.actions)

    def screenshot_game_area(self):
        x, y, width, height = 0, 25, 1280, 828
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        return screenshot, x, y

    PIXEL_TOL = 4  # “how close” counts as touching

    def _poly_touches_box(self, poly, box, tol=PIXEL_TOL) -> bool:
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
            self,
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
                    if self._poly_touches_box(poly, box)
                ]
                if len(touched) >= 2:  # segment links ≥2 stations
                    for i in range(len(touched)):
                        for j in range(i + 1, len(touched)):
                            u, v = touched[i], touched[j]
                            # MultiGraph ⇒ each call adds a *new* edge
                            G.add_edge(u, v, colour=colour)

        return G

    def perceive(self):
        """
        Capture a fresh screenshot → run detectors → build a symbolic observation.
        """
        self.screenshot, self.x0, self.y0 = self.screenshot_game_area()
        # ↓ helper now returns: (annot_img, bboxes, shapes)
        annot_stations, boxes, station_shapes = annotate_stations(self.screenshot)

        # ↓ helper now returns: (annot_img, polygons, colours)
        annot_lines, polygons, line_colours = annotate_lines(self.screenshot, conf=0.5, overlap=0.4)

        ctx, _ = classify_context(self.screenshot)

        graph = self.connect_stations(
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
                if self._poly_touches_box(poly, box)
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
                        if self._poly_touches_box([(vx, vy)], box):
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
        passengers = passenger_cnt(self.screenshot)
        self.world = {
            "graph": graph,
            "stations": stations,
            "unconnected": unconnected_stations,
            "segments": segments,
            "context": ctx,
            "passenger": passengers
        }





    def _compute_reward(self) -> float:
        """+1 the first time a specific pair is connected, else 0."""
        if not self.actions:
            return 0.0
        u, v = self.actions[-1]
        pair = tuple(sorted((u, v)))
        if pair in self.connected_pairs:
            return 0.0
        self.connected_pairs.add(pair)
        return 1.0

    def step(self, action_idx):
        """
        Parameters
        ----------
        action_idx : int
            Index into self.actions or however you want to encode actions.
        Returns
        -------
        obs_next : object        # the next observation
        reward   : float         # scalar reward
        done     : bool          # episode-termination flag
        info     : dict          # optional debug information
        """
        self._execute_action(action_idx)
        self.perceive()

        obs_next = self.world
        reward = self._compute_reward()
        done = self.world["context"] == "GAME_OVER"

        info = {}
        return obs_next, reward, done, info


    def reset(self):
            """
            Start a new episode.
            Returns
            -------
            obs : dict   the first observation
            """
            # take a fresh screenshot and (re-)build symbolic state
            self.perceive()
            return self.world




