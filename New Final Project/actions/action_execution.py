import math
import time
from lib2to3.pgen2.token import NEWLINE

import pyautogui
from typing import List, TYPE_CHECKING
from vision.perception import count_river_crossings
from actions.macro_definition import Verb, Speed, Reward, PoolItem, PAction
if TYPE_CHECKING:                # prevents runtime import loop
    from mini_metro_env import MiniMetroEnv
# -----------------------------------------------------------------------------
# MiniMetroEnv._execute_action
# -----------------------------------------------------------------------------
def execute_action(env: "MiniMetroEnv", action_idx: int) -> None:
    """
    Low-level bridge *RL-action → GUI action*.

    Parameters
    ----------
    env
    action_idx : int
        Index in `self.actions` produced by `_build_action_space()`.
    """
    if not (0 <= action_idx < env.action_space_n):
        print(f"[WARN] Invalid action id {action_idx}. No-op.")
        return

    act: PAction = env.actions[action_idx]

    # ────────────────────────────────────────────────────────────────────────
    #  Helper: absolute coordinates for various game objects
    # ────────────────────────────────────────────────────────────────────────
    def _station_xy(station_id: int) -> tuple[int, int]:
        """Return centre pixel of the given station."""
        cx, cy = env.world["stations"][station_id]["centre"]
        return env.x0 + cx, env.y0 + cy

    def _segment_xy(seg: dict) -> tuple[int, int]:
        """Mid-point of a line segment polygon."""
        poly = seg["poly"]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        return int(env.x0 + mx), int(env.y0 + my)

    # Coordinates of the four “speed buttons” / pool items etc.
    # Tune these once and reuse.

    SPEED_BUTTON = {
        Speed.PAUSE:   (1230, 111),
        Speed.UNPAUSE: (1230, 111),
        Speed.NORMAL:  (1230, 144),
        Speed.FASTER:  (1230, 175),
        Speed.CLOCK:   (1230, 67)
    }
    REWARD_BUTTON = {
        Reward.LEFT:   (519, 448),
        Reward.RIGHT:  (755, 445),
        Reward.CENTRE: (641, 450)
    }
    POOL_ITEM_BTN = {
        PoolItem.INTERCHANGE: (828, 1580),
        PoolItem.CARRIAGE:    (888, 1580),
        PoolItem.TRAIN:       (938, 1580),
    }

    # ────────────────────────────────────────────────────────────────────────
    #  Action dispatch
    # ────────────────────────────────────────────────────────────────────────
    if act.verb is Verb.NO_OP:
        return

    # 1️⃣ UI-only verbs ------------------------------------------------------
    if act.verb is Verb.TOGGLE_PAUSE:
        pyautogui.press("space")            # game default
        return

    if act.verb is Verb.SET_SPEED:
        xy = SPEED_BUTTON[Speed(act.arg)]
        pyautogui.click(*xy)
        return

    if act.verb is Verb.CHOOSE_REWARD:
        xy = REWARD_BUTTON[Reward(act.arg)]
        pyautogui.click(*xy)
        if act.arg == Reward.INTERCHANGE:
            env.world["assets"]["interchange"] += 1
        if act.arg == Reward.TRAIN:
            env.world["asset"]["train"] += 1
        if act.arg == Reward.CARRIAGE:
            env.world["asset"]["carriage"] += 1
        if act.arg == Reward.BRIDGE_TWO:
            env.world["asset"]["bridge"] += 2
        if act.arg == Reward.BRIDGE_ONE:
            env.world["asset"]["bridge"] += 1
        if act.arg == Reward.SHINKANSEN:
            env.world["asset"]["shinkansen"] += 2
        return

    # 2️⃣  Pool manipulation --------------------------------------------------
    if act.verb is Verb.PICK_POOL_ITEM:
        xy = POOL_ITEM_BTN[PoolItem(act.arg)]
        pyautogui.hold(*xy)
        return

    if act.verb is Verb.DRAG_POOL_ITEM:
        item, dst = act.arg, act.arg2
        src_xy = POOL_ITEM_BTN[PoolItem(item)]
        dst = env.world.get[dst]["centre"]
        # destination can be a station or a line segment
        if isinstance(dst, int):            # station id
            dst_xy = _station_xy(dst)
        else:                               # segment dict
            dst_xy = _segment_xy(dst)
        pyautogui.moveTo(*src_xy)
        pyautogui.dragTo(*dst_xy, duration=0.5, button="left")
        return

    # 3️⃣  Map interactions ---------------------------------------------------
    if act.verb is Verb.SELECT_STATION:
        pyautogui.hold(*(env.world[act.arg]["centre"]))
        return

    if act.verb is Verb.DISCONNECT_STATION:
        station_coord = env.world[act.arg]["centre"]
        line_coord = env.world[act.arg2]["endpoint"]
        pyautogui.moveTo(line_coord, button="left")
        pyautogui.dragTo(station_coord, button="left")
        return

    if act.verb is Verb.SELECT_LINE:
        pyautogui.hold(env.world[act.arg])
        return

    if act.verb is Verb.DRAG_TRACK:
        line_seg, station_id = act.arg, act.arg2
        src_xy = env.world[line_seg]["endpoint"]  # screen-coords
        dst_xy = env.world[station_id]["centre"]

        # Convert to *world* coords (same frame as river_mask)
        p1 = (src_xy[0] - env.x0, src_xy[1] - env.y0)
        p2 = (dst_xy[0] - env.x0, dst_xy[1] - env.y0)

        needed = count_river_crossings(p1, p2, env.world["river"])
        have = env.world["assets"]["bridge"]

        if needed > have:  # not enough tunnels/bridges
            print(f"[INFO] drag cancelled – need {needed} bridge(s), "
                  f"only {have} available.")
            return  # NO-OP

        # consume the resource first (defensive)
        env.world["assets"]["bridge"] -= needed

        # perform the GUI drag
        pyautogui.moveTo(*src_xy)
        pyautogui.dragTo(*dst_xy, duration=0.4, button="left")

        return

    if act.verb == NEWLINE:
        src = act.arg
        dst = act.arg2


    # 4️⃣  Fallback -----------------------------------------------------------
    #print(f"[TODO] verb {act.verb} not yet implemented.")