# plugins/SerpentMiniMetro_RL_AgentGameAgentPlugin/recorder.py
import csv, os
from time import time
from sneakysnek.recorder import Recorder
from sneakysnek.events    import KeyboardEvent, MouseEvent

class CSVActionRecorder:
    """
    Records global keyboard & mouse events to a CSV file that lines up with
    Serpent's frame timestamps (time.time()).
    """
    def __init__(self, path="datasets/inputs_mm.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file   = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["timestamp", "kind", "code", "x", "y"]
        )
        self.rec = Recorder.record(self._callback)

    # ------------------------------------------------------------------ #
    def _callback(self, event):
        if isinstance(event, KeyboardEvent):
            self.writer.writerow(
                [event.timestamp, "K_"+event.event.name, event.keyboard_key.name, "", ""]
            )
        elif isinstance(event, MouseEvent):
            self.writer.writerow(
                [event.timestamp, "M_"+event.event.name, event.button.name, event.x, event.y]
            )

    # ------------------------------------------------------------------ #
    def stop(self):
        self.rec.stop()
        self.file.close()
