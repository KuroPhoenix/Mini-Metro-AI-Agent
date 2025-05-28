import numpy as np

def MiniMetroState(agent, game_frame):
    # toy example – replace with your own logic
    stations = agent.sprite_locator.locate_multiple(
        list(agent.game.sprites.values()),
        game_frame.frame
    )
    # flatten (x,y) coords and pad to fixed length
    vec = np.zeros(100)
    for i, (_, (y0,x0,y1,x1)) in enumerate(stations.items()):
        vec[i*2:i*2+2] = [(x0+x1)/2, (y0+y1)/2]
    return vec.astype(np.float32) / 1280.0    # normalise to [0,1]

# plugins/SerpentMiniMetro_RL_AgentGameAgentPlugin/files/state_extractor.py

from typing import List, Dict
from .vision import Station, LineRoute

class WorldModel:
    """
    Keeps the latest game state that the vision system extracted so the
    RL policy (or hard-coded heuristics) can query it.
    """
    def __init__(self):
        self.stations:   List[Station]        = []
        self.routes:     LineRoute            = {}
        self.inventory:  Dict[str, int]       = {}
        self.weekly_offer: List[str]          = []

    # simple “last frame wins” update
    def update(self, *, stations=None, routes=None, inventory=None):
        if stations is not None:
            self.stations = stations
        if routes is not None:
            self.routes   = routes
        if inventory is not None:
            self.inventory = inventory
