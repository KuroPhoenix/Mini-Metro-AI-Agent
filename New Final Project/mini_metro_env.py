from __future__ import annotations

from actions.macro_definition import Verb, PAction, PoolItem, Speed, Reward
from actions.action_execution import execute_action
from actions.build_action_space import build_action_space
import reward_function as rt
from vision.perception import (
    screenshot_game_area,
    perceive,
    connect_stations,
)
import time
import pyautogui
import numpy as np
from pynput import mouse
from typing import Any, Callable, Tuple
import threading


class MiniMetroEnv:
    def __init__(self):
        self.world = {}
        self.last_action = None
        self.lines: dict = {}
        self.connected_pairs: set[tuple[int, int]] = set()
        self.assets = {
            "locomotive": 0,
            "carriage": 0,
            "bridge": 0,
            "line": 0,
            "interchange": 0,
            "shinkansen": 0
        }
        self.screenshot, self.x0, self.y0 = screenshot_game_area()
        self.perceive()
        self._refresh_actions()  # fills self.actions + self.action_space_n
        self.done = False

    def _refresh_actions(self) -> None:
        """(Re)compute the macro-action list and mirror it everywhere."""
        self.actions: list[PAction] = build_action_space(self.world)
        self.action_space_n = len(self.actions)
        self.world["actions"] = self.actions  # keep a copy in the world-dict

    def perceive(self) -> None:
        """Grab a new screenshot and update `self.world` in-place."""
        self.screenshot, self.world = perceive(self.assets)
        self._refresh_actions()  # action list may have changed after vision

    def _nearest_station(self, x, y) -> int:
        """Return the station id closest to pixel (abs coords)."""
        gx, gy = x - self.x0, y - self.y0  # convert to game coords
        stations = self.world["stations"]
        dists = {
            idx: (gx - c["centre"][0]) ** 2 + (gy - c["centre"][1]) ** 2
            for idx, c in stations.items()
        }
        if not dists:  # vision could not find any stations yet
            raise RuntimeError("No stations detected – run perceive() first?")
        return min(dists, key=dists.get)

    # ----------------------------------------------------------------------------
    #  UI hit‑boxes (screen pixel coords) – tweak once then forget
    # ----------------------------------------------------------------------------
    # These defaults work for a 1920×1080 full‑screen game window.  If your layout
    # is different, adjust the numbers or compute them at runtime.
    _PAUSE_BTN = (1860, 20, 1900, 60)
    _SPEED_BTNS = {
        Speed.PAUSE: (1700, 20, 1740, 60),
        Speed.NORMAL: (1745, 20, 1785, 60),
        Speed.FASTER: (1790, 20, 1830, 60),
    }
    _REWARD_BTNS = {
        Reward.LEFT: (760, 340, 960, 540),
        Reward.CENTRE: (960, 340, 1160, 540),
        Reward.RIGHT: (1160, 340, 1360, 540),
    }
    # pool slots: six icons bottom‑left; update if you use another skin
    _POOL_SLOTS = {i: (40 + i * 60, 1000, 100 + i * 60, 1060) for i in range(6)}

    # utility -------------------------------------------------------------

    def _inside(self, box: Tuple[int, int, int, int], x: int, y: int) -> bool:
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _euclid(self, p, q):
        return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5

    # ----------------------------------------------------------------------------
    #  Patch method – add to MiniMetroEnv
    # ----------------------------------------------------------------------------
    # ──────────────────────────────────────────────────────────────────
    #  Low-level helper: wait for a drag gesture
    # ──────────────────────────────────────────────────────────────────
    def _block_until_drag(self) -> dict[str, tuple[int, int]]:
        """
        Wait until the user performs a *left-button* drag and return the
        down/up positions, e.g. ``{"down": (x0, y0), "up": (x1, y1)}``.

        Implementation details
        ----------------------
        • Uses ``pynput.mouse.Listener`` to capture low-level events on
          all major OSes without the Tk dependency that `mouseinfo`
          pulls in.
        • A small Event is used to make the call blocking while keeping
          the listener in its own thread.
        """
        down_pos: tuple[int, int] | None = None
        up_pos:   tuple[int, int] | None = None
        done = threading.Event()

        def on_click(x: int, y: int, button: mouse.Button, pressed: bool):
            nonlocal down_pos, up_pos
            if button is not mouse.Button.left:
                return None

            if pressed and down_pos is None:             # mouse-down
                down_pos = (x, y)
                return None

            elif not pressed and down_pos is not None:   # mouse-up
                up_pos = (x, y)
                done.set()                               # unblock the main thread
                return False                             # stop listener
            return None

        # Start listener in a context-manager so it cleans up automatically
        with mouse.Listener(on_click=on_click) as listener:
            done.wait()          # blocks until Event is set in on_click

        # At this point the listener has stopped and coordinates are set
        assert down_pos is not None and up_pos is not None, "drag coords missing"
        return {"down": down_pos, "up": up_pos}

    def wait_for_human_action(self) -> int:
        """Block until the human performs a *click* or *drag* and map it to
        the corresponding PAction index in `self.actions`.  Supports every verb
        emitted by build_action_space.py.
        """
        # ------------------------------------------------------------------
        # 1. Raw gesture from OS
        # ------------------------------------------------------------------
        evt = self._block_until_drag()  # {'down':(x0,y0),'up':(x1,y1)}
        down, up = evt["down"], evt["up"]
        is_click = self._euclid(down, up) < 8  # <8 px ⇒ treat as click

        # ------------------------------------------------------------------
        # 2. Always start with a *fresh* symbolic world + action list
        # ------------------------------------------------------------------
        if not self.world.get("stations"):
            self.perceive()
        self._refresh_actions()

        # helper to locate the first index satisfying predicate ----------------
        def _find(pred: Callable[[Any], bool]) -> int | None:
            for idx, act in enumerate(self.actions):
                if pred(act):
                    return idx
            return None

        # ------------------------------------------------------------------
        # 3. CLICK – buttons, pool pick, selection
        # ------------------------------------------------------------------
        if is_click:
            x, y = down
            # Pause toggle --------------------------------------------------
            if self._inside(self._PAUSE_BTN, x, y):
                idx = _find(lambda a: isinstance(a, PAction) and a.verb is Verb.TOGGLE_PAUSE)
                if idx is not None:
                    return idx

            # Speed buttons -------------------------------------------------
            for spd, box in self._SPEED_BTNS.items():
                if self._inside(box, x, y):
                    idx = _find(lambda a: isinstance(a, PAction) and a.verb is Verb.SET_SPEED and a.arg is spd)
                    if idx is not None:
                        return idx

            # Weekly reward popup -------------------------------------------
            for choice, box in self._REWARD_BTNS.items():
                if self._inside(box, x, y):
                    idx = _find(lambda a: isinstance(a, PAction) and a.verb is Verb.CHOOSE_REWARD and a.arg is choice)
                    if idx is not None:
                        return idx

            # Pool *pick* ---------------------------------------------------
            for pid, box in self._POOL_SLOTS.items():
                if self._inside(box, x, y):
                    idx = _find(
                        lambda a: isinstance(a, PAction) and a.verb is Verb.PICK_POOL_ITEM and a.arg is PoolItem(pid))
                    if idx is not None:
                        return idx

            # Station / line *selection* ------------------------------------
            st = self._nearest_station(x, y)
            if st is not None:
                idx = _find(lambda a: isinstance(a, PAction) and a.verb is Verb.SELECT_STATION and a.arg == st)
                if idx is not None:
                    return idx
            ln = getattr(self, "_nearest_line", lambda *_: None)(x, y, 20)
            if ln is not None:
                idx = _find(lambda a: isinstance(a, PAction) and a.verb is Verb.SELECT_LINE and a.arg == ln)
                if idx is not None:
                    return idx

        # ------------------------------------------------------------------
        # 4. DRAG – track or pool‑item drag
        # ------------------------------------------------------------------
        start_st = self._nearest_station(*down)
        end_st = self._nearest_station(*up)

        # Track drag station → station -------------------------------------
        if start_st is not None and end_st is not None:
            idx = _find(lambda a: isinstance(a,
                                             PAction) and a.verb is Verb.DRAG_TRACK and a.arg == start_st and a.arg2 == end_st)
            if idx is not None:
                return idx

        # Pool‑item drag ----------------------------------------------------
        pool_id = None
        for pid, box in self._POOL_SLOTS.items():
            if self._inside(box, *down):
                pool_id = PoolItem(pid)
                break
        if pool_id is not None:
            # → station
            if end_st is not None:
                idx = _find(lambda a: isinstance(a,
                                                 PAction) and a.verb is Verb.DRAG_POOL_ITEM and a.arg == pool_id and a.arg2 == end_st)
                if idx is not None:
                    return idx
            # → segment (line body)
            seg = getattr(self, "_nearest_segment", lambda *_: None)(*up, 15)
            if seg is not None:
                idx = _find(lambda a: isinstance(a,
                                                 PAction) and a.verb is Verb.DRAG_POOL_ITEM and a.arg == pool_id and a.arg2 == seg)
                if idx is not None:
                    return idx

        # ------------------------------------------------------------------
        # 5. Fallback – no valid action found
        # ------------------------------------------------------------------
        print("[Env] Human gesture not in action space; try again.")
        # Refresh and recurse (rare)
        self._refresh_actions()
        return self.wait_for_human_action()

    def action_index(self, station_A: int, station_B: int) -> int:
        """Map a (station, station) pair to the corresponding DRAG_TRACK action."""
        for i, act in enumerate(self.actions):
            if act.verb == Verb.DRAG_TRACK:
                u, v = act.arg  # that helper stores a tuple
                if {u, v} == {station_A, station_B}:
                    return i
        # fallback → NO-OP
        return next(i for i, a in enumerate(self.actions) if a.verb is Verb.NO_OP)

    def _compute_reward(self) -> float:
        prev = getattr(self, "prev_passengers", 0)
        # total_reward returns (reward, new_prev_passengers)
        rew, new_prev = rt.total_reward(
                self.world,
                prev_passengers = prev,
        )

        # stash for next time and return scalar
        self.prev_passengers = new_prev
        return rew

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
        execute_action(self, action_idx)
        self.last_action = self.actions[action_idx]
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
            self.connected_pairs.clear()
            obs_next = self.world
            reward = self._compute_reward()
            done = self.world["context"] == "GAME_OVER"

            info = {}
            return obs_next, reward, done, info




