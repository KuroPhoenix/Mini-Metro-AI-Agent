# actions/build_action_space.py
"""Action‑space generator for **Mini Metro RL**

This supersedes the stub that only produced a handful of actions.
It now enumerates **every verb currently implemented in
``actions.action_execution`` / ``actions.macro_definition``** so that the
agent can explore the full GUI.

The function is *defensive* – if some detectors (vision) are still
missing data, we silently skip any branch that relies on that data
instead of raising.

Returned objects are the dataclass ``PAction`` so they work directly with
``MiniMetroEnv`` + ``execute_action``.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Any

from actions.macro_definition import (
    Verb,
    Speed,
    PoolItem,
    Reward,
    PAction,
)

__all__ = ["build_action_space"]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _coerce_world(world: Any) -> Dict:
    """Accept either the raw *world* dict **or** the whole ``MiniMetroEnv``.

    ``MiniMetroEnv`` wraps the dict in ``env.world`` – unwrap it so callers
    can be sloppy.
    """
    if hasattr(world, "world"):
        return world.world  # type: ignore[attr-defined]
    if not isinstance(world, dict):
        raise TypeError("world must be a dict or MiniMetroEnv instance")
    return world


def _index_segments(world: Dict) -> List[int]:
    """Ensure every segment is accessible via ``world[idx]``.

    ``action_execution`` fetches segment‑dicts with
    ``env.world.get(line_seg)['endpoint']``.  We therefore alias each entry
    in ``world['segments']`` to an *integer* key at the top level, which is
    fast and avoids allocating new IDs on every frame.
    """
    seg_list = world.get("segments") or []
    for idx, seg in enumerate(seg_list):
        if idx not in world:  # create once – cheap no‑op in subsequent frames
            world[idx] = seg
    return list(range(len(seg_list)))


# Mapping from asset‑string → PoolItem enum
_ASSET_TO_POOL = {
    "locomotive": PoolItem.TRAIN,
    "carriage": PoolItem.CARRIAGE,
    "interchange": PoolItem.INTERCHANGE,
    "shinkansen": PoolItem.SHINKANSEN,
}


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def build_action_space(world) -> List[PAction]:
    """Return a complete list of legal *high‑level* actions for the current frame."""

    world = _coerce_world(world)

    # Unpack the vision state -------------------------------------------------
    stations: Dict[int, Dict] = world.get("stations", {})  # id → dict
    lines: Dict[int, Dict] = world.get("lines", {})        # id → dict
    segments: List[int] = _index_segments(world)           # integer keys just created
    assets: Dict[str, int] = world.get("assets", {})

    actions: List[PAction] = []

    # -------------------------------------------------------------------
    # 0️⃣  Meta / speed controls
    # -------------------------------------------------------------------
    actions.append(PAction(Verb.TOGGLE_PAUSE))
    for spd in Speed:
        actions.append(PAction(Verb.SET_SPEED, spd))

    # -------------------------------------------------------------------
    # 1️⃣  Map interactions – select entities / drag track / disconnect
    # -------------------------------------------------------------------
    for st_id in stations.keys():
        actions.append(PAction(Verb.SELECT_STATION, st_id))

    for ln_id in lines.keys():
        actions.append(PAction(Verb.SELECT_LINE, ln_id))

    # Drag new track from a line‑segment endpoint → a *different* station
    for seg_id in segments:
        seg = world.get(seg_id, {})
        touched = set(seg.get("stations", []))
        for st_id in stations.keys():
            if st_id in touched:
                continue
            actions.append(PAction(Verb.DRAG_TRACK, seg_id, st_id))

    # Disconnect a station from its segment (both directions)
    for seg_id in segments:
        seg = world.get(seg_id, {})
        for st_id in seg.get("stations", []):
            actions.append(PAction(Verb.DISCONNECT_STATION, st_id, seg_id))

    for station in stations:
        for station_dst in stations:
            actions.append(PAction(Verb.NEW_LINE, station, station_dst))
    # -------------------------------------------------------------------
    # 2️⃣  Pool manipulation – pick and drag upgrades
    # -------------------------------------------------------------------
    for asset_name, count in assets.items():
        if count <= 0:
            continue  # nothing left of this item
        pool_enum: Optional[PoolItem] = _ASSET_TO_POOL.get(asset_name)
        if pool_enum is None:
            continue  # unrecognised asset key

        # pick (left‑click to lift into hand)
        actions.append(PAction(Verb.PICK_POOL_ITEM, pool_enum))

        # drag onto every station
        for st_id in stations.keys():
            actions.append(PAction(Verb.DRAG_POOL_ITEM, pool_enum, st_id))

        # drag onto every segment (line) – symmetric with DRAG_TRACK dest rules
        for seg_id in segments:
            actions.append(PAction(Verb.DRAG_POOL_ITEM, pool_enum, seg_id))

    # -------------------------------------------------------------------
    # 3️⃣  Weekly reward choice (if the dialogue is up – agent can ignore)
    # -------------------------------------------------------------------
    for rwd in Reward:
        actions.append(PAction(Verb.CHOOSE_REWARD, rwd))

    # -------------------------------------------------------------------
    # 4️⃣  Fallback – always allow NO‑OP so the agent can wait a tick
    # -------------------------------------------------------------------
    actions.append(PAction(Verb.NO_OP))

    return actions
