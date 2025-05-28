from pathlib import Path

from serpent.game_agent import GameAgent
from serpent.input_controller import MouseButton
from serpent.machine_learning.context_classification.context_classifiers \
    import cnn_inception_v3_context_classifier as _cc
from serpent.machine_learning.context_classification \
    .context_classifiers.cnn_inception_v3_context_classifier import \
    CNNInceptionV3ContextClassifier

from .recorder import CSVActionRecorder
from .vision import detect_stations, detect_lines, colour_masks, Color, detect_inventory, detect_weekly_offer
from fastai.basic_train import load_learner
from .state_extractor import WorldModel
import importlib.util, sys
import importlib
import cv2, numpy as np
from fastai.vision.image import Image
from fastai.vision import pil2tensor
import PIL


# save the old method
_orig_predict = CNNInceptionV3ContextClassifier.predict

def _fastai_predict(self, input_frame):
    # if this classifier has a FastAI learner underneath…
    if hasattr(self, "classifier") and hasattr(self.classifier, "data"):
        # 1) BGR→RGB → PIL
        rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        pil  = PIL.Image.fromarray(rgb)
        # 2) PIL→tensor→normalized fastai Image
        fa_img = Image(pil2tensor(pil, dtype=np.float32).div_(255.0))
        # 3) call the FastAI learner
        _,_,probs = self.classifier.predict(fa_img)
        return probs.numpy()
    # otherwise (Keras path) fall back
    return _orig_predict(self, input_frame)

# overwrite the method on the class
CNNInceptionV3ContextClassifier.predict = _fastai_predict

_orig = _cc.CNNInceptionV3ContextClassifier.load_classifier

def load_module_from_path(mod_name: str, file_path: Path):
    "Return a module object whose __file__ is `file_path`."
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(file_path),
        submodule_search_locations=[str(file_path.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    # ⬇️  make the package visible **before** its code runs
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module



def _patched(self, file_path):
    p = Path(file_path)
    if p.suffix == ".pkl":                 # fastai export
        self.classifier = load_learner(p.parent, p.name)   # <-- key change
        self.classifier.model.eval()

    else:                                  # keras .model / .h5
        _orig(self, str(p))

_cc.CNNInceptionV3ContextClassifier.load_classifier = _patched



def _draw_box(img, xmin,ymin,xmax,ymax, text):
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
    cv2.putText(img, text, (xmin, ymin-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return img


class SerpentMiniMetro_RL_AgentGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        # register new frame handler
        self.frame_handler_setups["learn"] = self.setup_learn
        self.frame_handlers["learn"] = self.handle_learn

    def setup_play(self):
        self.context_classifier = CNNInceptionV3ContextClassifier(
            input_shape=(384, 512, 3)
        )

        base = Path(__file__).resolve()

        plugins_root = base.parents[3]          # …/plugins
        candidates = [
            base.parent / "ml_models",  # agent’s own models
            base.parents[2] / "SerpentMiniMetroGamePlugin" / "files" / "ml_models",
        ]

        for root in candidates:
            for fname in ("context_classifier_InceptionV3.model",
                          "context_classifier_fa_m0_learner.pkl"):
                model_path = root / fname
                print("[DEBUG] checking", model_path)
                if model_path.is_file():
                    self.context_classifier.load_classifier(model_path)
                    # build a class-name list once
                    print(f"[INIT] loaded {model_path.name}")
                    # -----------------------------------------------------------------
                    # Fast-ai Learner:             self.classifier.data.classes
                    # Keras model you trained:     give your own list (fallback)
                    # -----------------------------------------------------------------
                    try:
                        self.class_names = list(self.context_classifier.classifier.data.classes)
                    except AttributeError:  # Keras path
                        self.class_names = [
                            "splash", "main_menu", "in_game", "pause", "game_over", "other"
                        ]
                    print(f"[INIT] loaded {model_path.name} with {len(self.class_names)} classes")
                    break
            else:
                continue
            break
        else:
            raise FileNotFoundError(
                f"No model found in: {[str(r) for r in candidates]}"
            )

        self.world = WorldModel()


    def handle_play(self, game_frame):
        """
        Runs at the full Serpent capture FPS (60 Hz by default).
        We’ll do three tiny things:
          1. push the frame into the VisualDebugger (“raw” tab)
          2. run the context classifier and print it
          3. left-click once when the first in-game frame appears
        """
        frame_bgr = game_frame.frame  # already BGR on macOS
        annot = frame_bgr.copy()
        routes = detect_lines(frame_bgr)
        inv = detect_inventory(frame_bgr)
        stations = detect_stations(frame_bgr, debug_img=annot)

        self.world.update(stations=stations, routes = routes, inventory = inv)

        if self._looks_like_weekly(frame_bgr):
                self.world.weekly_offer = detect_weekly_offer(frame_bgr)
        if self.world.weekly_offer:
                print("▶ Weekly offer:", self.world.weekly_offer)

        #/ *draw * /
        for st in stations:
            cv2.circle(annot, st.center, st.radius, (0, 0, 255), 2)
        for color, mask in colour_masks(frame_bgr).items():
            annot[mask > 0] = (0, 255, 0) if color == Color.GREEN else annot[mask > 0]
        for st in stations:
            text = st.shape.name  # e.g. "TRIANGLE", "DIAMOND", etc.
            x, y = st.center
            cv2.putText(annot, text, (x + st.radius, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


        # 2️⃣ send *both* the raw and annotated frames
        self.visual_debugger.store_image_data(
            image_data=frame_bgr, image_shape=frame_bgr.shape, bucket="raw")
        self.visual_debugger.store_image_data(
            image_data=annot, image_shape=annot.shape, bucket="det")

        rgb = cv2.cvtColor(game_frame.frame, cv2.COLOR_BGR2RGB)
        pil = PIL.Image.fromarray(rgb)

        # 3. predict
        probs = self.context_classifier.predict(frame_bgr)
        # 2️⃣  classify and print
        label_idx = int(np.argmax(probs))
        label = self.class_names[label_idx]

        # only print when the label changes
        if label != getattr(self, "_ctx_prev", None):
            print(f"[CTX] {label}  (p={probs[label_idx]:.2f})")
            self._ctx_prev = label

        # 3️⃣  demo click when the agent first sees "in_game"
        if label == "in_game" and not getattr(self, "_clicked", False):
            self.input_controller.click(200, 200, button=MouseButton.LEFT)
            self._clicked = True

      # ------------------------------------------------------------
      # helper : screen gets ~25 % darker when Monday popup shows


    def _looks_like_weekly(self, frame_bgr) -> bool:
            v = np.mean(frame_bgr[:, :, 2])  # red channel as proxy
            if not hasattr(self, "_ref_brightness"):
                self._ref_brightness = v
                return False
            return v < self._ref_brightness * 0.75
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

    # RL & vision helpers
    # from stable_baselines3 import PPO                  # or your choice


