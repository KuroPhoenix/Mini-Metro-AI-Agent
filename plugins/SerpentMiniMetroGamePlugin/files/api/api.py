from pathlib import Path

from serpent.game_api import GameAPI
from .image_classifier import image_classifier
#from .object_detector.pytorch_detector import PytorchObjectDetector as object_detector

import offshoot


class MiniMetroAPI(GameAPI):

    def __init__(self, game=None):
        super().__init__(game=game)
        # either from config, or fall back to the plugin’s own ml_models folder
        try:
            self.ml_models_dir = Path(offshoot.config['file_paths']['game_ml_models'])
        except KeyError:
            # __file__ → .../files/api/api.py → parent.parent → files/
            self.ml_models_dir = Path(__file__).resolve().parent.parent / "ml_models"

        """ self.ml_station_detector = object_detector(
            model_path=self.ml_models_dir/'station_detector_tf_m6-1_frozen_graph.pb',
            model_type='tensorflow',
            category_labels_path=self.ml_models_dir/'station_detector_label_map.json'
        ) """
        self.ml_station_detector = None
        self.ml_context_classifier = image_classifier(
            model_path=self.ml_models_dir/'context_classifier_fa_m0_learner.pkl',
            model_type='fastai'
        )

    def parse_game_state(self, frame):
        pass
