"""reward_function.py – robust, dict‑friendly version
=================================================
This version accepts **either** the dict you build in `vision.perception()`
*or* the original object‑oriented `world` that the public Mini‑Metro AI
examples use.  All keys/attributes that might be missing early in a run
safely default to 0 or an empty iterable, so nothing crashes while the
screen‑parsers are still populating state.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence, Any, Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 🧩  Compatibility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_world(world: Any) -> Dict[str, Any]:
    """Return a **dict** no matter what representation the caller passes."""
    if isinstance(world, dict):
        return world

    # Dynamically expose the attributes we rely on.
    w: Dict[str, Any] = {
        "passengers_delivered": getattr(world, "passengers_delivered", 0),
        "map_bbox": getattr(world, "map_bbox", (0.0, 0.0, 1.0, 1.0)),
        "lines": {i: ln for i, ln in enumerate(getattr(world, "lines", []))},
        "stations": {i: st for i, st in enumerate(getattr(world, "stations", []))},
        "assets": getattr(world, "assets", {}),
    }
    return w


def _iter_lines(world_dict: Dict[str, Any]):
    return world_dict.get("lines", {}).values()


def _iter_stations(world_dict: Dict[str, Any]):
    return world_dict.get("stations", {}).values()


def _get(item: Any, attr: str, default=None):
    """Uniform access for dicts *and* objects."""
    if isinstance(item, dict):
        return item.get(attr, default)
    return getattr(item, attr, default)

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helper
# ─────────────────────────────────────────────────────────────────────────────

def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ─────────────────────────────────────────────────────────────────────────────
# Passenger throughput
# ─────────────────────────────────────────────────────────────────────────────

def passenger_throughput(prev_passengers: int, current_passengers: int, *, lam: float = 1.0) -> Tuple[float, int]:
    delta = current_passengers - prev_passengers
    return lam * float(delta), current_passengers

# ─────────────────────────────────────────────────────────────────────────────
# Unused‑asset penalty
# ─────────────────────────────────────────────────────────────────────────────

def asset_usage(world, *, lam: float = 0.5) -> float:
    w = _unwrap_world(world)
    unused = 0
    for line in _iter_lines(w):
        unused += _get(line, "carriages", 0) + _get(line, "locomotives", 0)
    return -lam * float(unused)

# ─────────────────────────────────────────────────────────────────────────────
# Shape diversity
# ─────────────────────────────────────────────────────────────────────────────

def _diversity_bonus_for_line(line) -> float:
    st = _get(line, "stations", [])
    bonus = 0.0
    for a, b, c in zip(st, st[1:], st[2:]):
        shapes = {_get(a, "shape"), _get(b, "shape"), _get(c, "shape")}
        shapes.discard(None)
        if len(shapes) == 3:
            bonus += 1.0
        elif len(shapes) == 1 and shapes:
            bonus -= 1.0
    return bonus


def diversity_reward(world, *, lam: float = 0.3) -> float:
    w = _unwrap_world(world)
    raw = sum(_diversity_bonus_for_line(L) for L in _iter_lines(w))
    n_triples = sum(max(0, len(_get(L, "stations", [])) - 2) for L in _iter_lines(w))
    return lam * (raw / n_triples) if n_triples else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Line‑span penalty (keep lines local)
# ─────────────────────────────────────────────────────────────────────────────

def span_penalty(world, *, lam: float = 3.0, theta: float = 0.45) -> float:
    w = _unwrap_world(world)
    xmin, ymin, xmax, ymax = w.get("map_bbox", (0, 0, 1, 1))
    D = max(xmax - xmin, ymax - ymin)
    penalty = 0.0
    for L in _iter_lines(w):
        stations = _get(L, "stations", [])
        if len(stations) < 2:
            continue
        d = dist(_get(stations[0], "pos", (0, 0)), _get(stations[-1], "pos", (0, 0))) / max(D, 1e-9)
        penalty += max(0.0, d - theta)
    return -lam * penalty

# ─────────────────────────────────────────────────────────────────────────────
# Quadrant confinement
# ─────────────────────────────────────────────────────────────────────────────

def _quadrant_id(x: float, y: float, *, midx: float, midy: float) -> int:
    return (x > midx) + 2 * (y > midy)


def quadrant_reward(world, *, lam: float = 1.0, k: float = 0.75) -> float:
    w = _unwrap_world(world)
    xmin, ymin, xmax, ymax = w.get("map_bbox", (0, 0, 1, 1))
    midx, midy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    score = 0.0
    for L in _iter_lines(w):
        stations = _get(L, "stations", [])
        if not stations:
            continue
        counts = Counter(_quadrant_id(*_get(s, "pos", (0, 0)), midx=midx, midy=midy) for s in stations)
        frac = max(counts.values()) / len(stations)
        if frac >= k:
            score += 1.0
    return lam * score

# ─────────────────────────────────────────────────────────────────────────────
# Special‑station coverage
# ─────────────────────────────────────────────────────────────────────────────
SPECIAL_SHAPES = {"★", "◆", "✚", "⬣"}


def special_coverage(world, *, lam: float = 2.0) -> float:
    w = _unwrap_world(world)
    reward = 0.0
    for st in _iter_stations(w):
        if _get(st, "shape") not in SPECIAL_SHAPES:
            continue
        touching = sum(st in _get(L, "stations", []) for L in _iter_lines(w))
        reward += 1.0 if touching >= 2 else -1.0
    return lam * reward

# ─────────────────────────────────────────────────────────────────────────────
# Straightness vs. zig‑zag
# ─────────────────────────────────────────────────────────────────────────────

def _straightness_for_line(line, zeta_good: float = 1.12, zeta_bad: float = 1.5) -> float:
    stations = _get(line, "stations", [])
    score = 0.0
    for a, b, c in zip(stations, stations[1:], stations[2:]):
        poly = dist(_get(a, "pos", (0, 0)), _get(b, "pos", (0, 0))) + dist(_get(b, "pos", (0, 0)), _get(c, "pos", (0, 0)))
        straight = dist(_get(a, "pos", (0, 0)), _get(c, "pos", (0, 0)))
        ratio = poly / straight if straight > 1e-9 else 1.0
        if ratio <= zeta_good:
            score += 1.0
        elif ratio > zeta_bad:
            score -= 1.0
    return score


def straightness_reward(world, *, lam: float = 0.3) -> float:
    w = _unwrap_world(world)
    raw = sum(_straightness_for_line(L) for L in _iter_lines(w))
    n_triples = sum(max(0, len(_get(L, "stations", [])) - 2) for L in _iter_lines(w))
    return lam * (raw / n_triples) if n_triples else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Loop reward
# ─────────────────────────────────────────────────────────────────────────────

def _is_loop(line) -> bool:
    st = _get(line, "stations", [])
    return len(set(st)) < len(st)


def loop_reward(world, *, lam: float = 1.5, density_thresh: int = 6, radius: float = 80.0) -> float:
    w = _unwrap_world(world)
    reward = 0.0
    all_positions = [_get(st, "pos", (0, 0)) for st in _iter_stations(w)]
    for L in _iter_lines(w):
        if not _is_loop(L):
            continue
        positions = [_get(st, "pos", (0, 0)) for st in _get(L, "stations", [])]
        if not positions:
            continue
        cx = sum(x for x, _ in positions) / len(positions)
        cy = sum(y for _, y in positions) / len(positions)
        nearby = sum(dist((cx, cy), p) <= radius for p in all_positions)
        if nearby >= density_thresh:
            reward += 1.0
    return lam * reward

# ─────────────────────────────────────────────────────────────────────────────
# Combine everything
# ─────────────────────────────────────────────────────────────────────────────

def total_reward(
    world,
    *,
    prev_passengers: int,
    lambda_dict: Dict[str, float] | None = None,
) -> Tuple[float, int]:
    """Add up every term with optional λ‑overrides."""
    λ = lambda_dict or {}
    w = _unwrap_world(world)

    curr = w.get("passengers_delivered", 0)
    r_pass, new_prev = passenger_throughput(prev_passengers, curr, lam=λ.get("pass", 1.0))

    components = {
        "pass": r_pass,
        "assets": asset_usage(w, lam=λ.get("asset", 0.5)),
        "div": diversity_reward(w, lam=λ.get("div", 0.3)),
        "span": span_penalty(w, lam=λ.get("span", 3.0)),
        "quad": quadrant_reward(w, lam=λ.get("quad", 1.0)),
        "special": special_coverage(w, lam=λ.get("special", 2.0)),
        "straight": straightness_reward(w, lam=λ.get("straight", 0.3)),
        "loop": loop_reward(w, lam=λ.get("loop", 1.5)),
    }

    reward = sum(components.values())
    return reward, new_prev


# ─────────────────────────────────────────────────────────────────────────────
# Public re‑exports for tidy imports elsewhere
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "passenger_throughput",
    "asset_usage",
    "diversity_reward",
    "span_penalty",
    "quadrant_reward",
    "special_coverage",
    "straightness_reward",
    "loop_reward",
    "total_reward",
]
