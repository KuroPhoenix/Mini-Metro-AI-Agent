# mini_metro/actions/space_builder.py
from typing import List, TYPE_CHECKING
from macro_definition import PAction, Verb, Speed, Reward, PoolItem

if TYPE_CHECKING:                # prevents runtime import loop
    from mini_metro_env import MiniMetroEnv


def build_action_space(env: "MiniMetroEnv") -> List[PAction]:
    acts: List[PAction] = []

    # Always-available verbs
    acts.extend(PAction(Verb.SET_SPEED, s) for s in Speed)
    acts.extend(PAction(Verb.CHOOSE_REWARD, r) for r in Reward)
    acts.extend(PAction(Verb.PICK_POOL_ITEM, p) for p in PoolItem)

    stations = env.world.get("stations", [])
    for station in stations:
        acts.append(PAction(Verb.SELECT_STATION, station))
        acts.append(PAction(Verb.DRAG_POOL_ITEM, PoolItem.INTERCHANGE, station))

    line_segments = env.world.get("segments", [])
    for line in line_segments:
        for item in PoolItem:
            if item is not PoolItem.INTERCHANGE:
                acts.append(PAction(Verb.DRAG_POOL_ITEM, item, line))
        acts.append(PAction(Verb.SELECT_LINE, line))
        for station in stations:
            acts.append(PAction(Verb.DRAG_TRACK, line, station))
            acts.append(PAction(Verb.DISCONNECT_STATION, station, line))

    return acts
