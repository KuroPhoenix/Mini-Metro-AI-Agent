from serpent.game import Game

from .api.api import MiniMetroAPI

from serpent.utilities import Singleton

import time


class SerpentMiniMetroGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "steam"

        kwargs["window_name"] = "Mini Metro"

        kwargs["app_id"] = "287980"
        kwargs["app_args"] = None
        
        
        

        super().__init__(**kwargs)

        self.api_class = MiniMetroAPI
        self.api_instance = None

    @property
    def screen_regions(self):
        regions = {
            "SAMPLE_REGION": (0, 0, 0, 0)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets

    def after_launch(self):
        """
        Wait until macOS has created window 1, then run the normal
        after_launch.  If moving the window raises -1719 we still
        record geometry so the frame-grabber has numbers to use.
        """
        time.sleep(2.0)        # increase if Mini Metro is slow to appear

        try:
            super().after_launch()      # find window → sets self.window_id
        except Exception:
            # swallow the AppleScript move-window error
            pass
        finally:
            # Ensure window_geometry exists
            if not self.window_geometry:
                try:
                    self.update_window_geometry()
                except Exception:
                    # last-ditch: fall back to the values you saw printed
                    self.window_geometry = {
                        "width": 1280,
                        "height": 828,
                        "x_offset": 0,
                        "y_offset": 25      # menu bar height on your MacBook
                    }
