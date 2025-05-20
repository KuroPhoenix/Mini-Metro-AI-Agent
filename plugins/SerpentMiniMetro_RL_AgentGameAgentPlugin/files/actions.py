# plugins/SerpentMiniMetro_RL_AgentGameAgentPlugin/files/actions.py
from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Optional


# ────────────────────────────────────────────────────────────────
# 1.1  VERBS  (the “what to do” channel)
# ────────────────────────────────────────────────────────────────
class Verb(IntEnum):
    # menu / meta
    PRESS_PLAY        = 0
    NAVIGATE_MAP      = 1          # arg = +1 / –1 page, or index
    PRESS_RESTART     = 2
    PRESS_MAIN_MENU   = 3

    # time controls
    TOGGLE_PAUSE      = 4
    SET_SPEED         = 5          # arg = Speed enum

    # station-level actions
    SELECT_STATION    = 6          # arg = StationShape enum
    SELECT_LINE_END   = 7          # arg = StationShape enum
    DRAG_TRACK        = 8          # arg = StationShape enum (target)

    # pool & upgrades
    PICK_POOL_ITEM    = 9          # arg = PoolItem enum
    DRAG_POOL_ITEM    = 10         # arg = PoolItem enum  (onto line / station)
    CHOOSE_REWARD     = 11         # arg = Reward enum

    NO_OP             = 12         # “do nothing” this frame


# ────────────────────────────────────────────────────────────────
# 1.2 ARG ENUMS  (the “details” channel)
# ────────────────────────────────────────────────────────────────
class Speed(IntEnum):
    NORMAL  = 0                # your “speed-one”
    FASTER  = 1                  # your “speed-two”


class StationShape(IntEnum):
    CIRCLE   = 0
    TRIANGLE = 1
    SQUARE   = 2
    STAR     = 3
    DIAMOND  = 4
    CROSS    = 5
    GEM      = 6
    PENTAGON = 7
    OVAL     = 8
    WEDGE    = 9
    # add more if DLC introduces them

class InterchangeStationShape(IntEnum):
    CIRCLE   = 0
    TRIANGLE = 1
    SQUARE   = 2
    STAR     = 3
    DIAMOND  = 4
    CROSS    = 5
    GEM      = 6
    PENTAGON = 7
    OVAL     = 8
    WEDGE    = 9

class PoolItem(IntEnum):
    TRAIN        = 0
    CARRIAGE     = 1
    HUB          = 2
    SHINKANSEN   = 3


class Reward(IntEnum):
    TRAIN        = 0
    CARRIAGE     = 1
    HUB          = 2
    SHINKANSEN   = 3
    NEW_LINE     = 4
    BRIDGE_ONE   = 5
    BRIDGE_TWO   = 6


# ────────────────────────────────────────────────────────────────
# 1.3 PARAMETRISED ACTION  (the object passed around)
# ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PAction:          # “parameterised action”
    verb: Verb
    arg: Optional[int] = None      # use the matching enum above
