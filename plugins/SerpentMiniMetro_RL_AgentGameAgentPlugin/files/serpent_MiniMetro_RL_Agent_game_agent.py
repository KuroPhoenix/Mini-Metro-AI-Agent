from serpent.game_agent import GameAgent
from .recorder import CSVActionRecorder
from actions import Verb, PAction, Speed, Reward, StationShape, InterchangeStationShape, PoolItem

class SerpentMiniMetro_RL_AgentGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        # register new frame handler
        self.frame_handler_setups["learn"] = self.setup_learn
        self.frame_handlers["learn"] = self.handle_learn

    def setup_play(self):
        pass

    def handle_play(self, game_frame):
        pass

    # ------------------------------------------------------------
    #  Called once before the first frame arrives
    def setup_learn(self):
        self.logger = CSVActionRecorder("datasets/inputs_mm.csv")

    # ------------------------------------------------------------
    #  Called every captured frame (2 fps in your config)
    def handle_learn(self, game_frame):
        """
        We don't control the game in this mode; we just let the recorder
        thread collect key/mouse events while Serpent keeps saving frames.
        """
        pass  # nothing to do per-frame

    # ------------------------------------------------------------
    def shutdown(self):
        """Serpent calls this when you press Q in the Serpent console."""
        if hasattr(self, "logger"):
            self.logger.stop()

    # def dispatch(self, p_action: PAction):
    #     """Turn a semantic PAction into concrete InputController calls."""
    #     v, a = p_action.verb, p_action.arg
    #
    #     if v is Verb.NO_OP:
    #         return
    #
    #     if v is Verb.SET_SPEED:
    #         btn = SPEED_BUTTON_COORDS[a]  # lookup table
    #         self.input_controller.click(*btn)
    #
    #     elif v is Verb.CHOOSE_REWARD:
    #         slot = self._which_slot_contains_reward(a)  # vision helper
    #         self.input_controller.click(*REWARD_SLOTS[slot])
    #
    #     elif v is Verb.SELECT_STATION:
    #         x, y = self._nearest_station_of_shape(a)  # from detector
    #         self.input_controller.click(x, y)
    #...and so on