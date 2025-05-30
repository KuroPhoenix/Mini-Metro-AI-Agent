import PIL
import cv2
import numpy as np
import enum
from typing import List, Tuple, Union
import pytesseract
from . import ocr   # ← your existing OCR wrapper

class Context(enum.IntEnum):
    IN_GAME     = 0       # No reward banner
    ONE_REWARD  = 1       # A single wide reward (e.g. “Locomotive” split into two halves)
    TWO_REWARDS = 2       # Two independent rewards shown side-by-side
    GAME_OVER   = 3

# ───────────────────────────────────────────────────────────────────────────
# 1.  Regions where the reward strings appear on your 1280×828 capture
# ────────────────────────────────────────────────────────────────────────────
_LEFT  = (402, 610, 245, 50)   # x, y, w, h
_RIGHT = (662, 614, 237, 50)
_PASSENGER = (1063, 57, 56, 21)
_GAME_OVER = (424, 119, 425, 100)
# ────────────────────────────────────────────────────────────────────────────
# 2.  Reward-name normalisation (OCR can be noisy, so accept common shards)
# ────────────────────────────────────────────────────────────────────────────
_REWARD_ALIASES = {
    "locomotive":  "TRAIN",
    "loco":        "TRAIN",
    "tive":        "TRAIN",
    "train":       "TRAIN",
    "carriage":    "CARRIAGE",
    "bridge":      "BRIDGE",   # adapt if you distinguish 1 vs 2
    "newline":     "NEW_LINE",
    "hub":         "HUB",
    "shinkansen":  "SHINKANSEN",
}


def _crop(frame: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
    """Return the BGR sub-image defined by region=(x, y, w, h)."""
    x, y, w, h = region
    return frame[y:y + h, x:x + w]


def _ocr(img: np.ndarray) -> str:
    """Run Tesseract on a small image patch and return raw text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(thresh).strip()


def _normalise(raw: str) -> str:
    """Collapse spaces, make lowercase – helps when matching aliases."""
    return raw.lower().replace(" ", "")


def _to_reward(raw: str) -> Union[str, None]:
    """Map OCR text to the canonical reward name; `None` if unrecognised."""
    key = _normalise(raw)
    for alias, canonical in _REWARD_ALIASES.items():
        if alias in key:
            return canonical
    return None


# ────────────────────────────────────────────────────────────────────────────
# 3.  Public API
# ────────────────────────────────────────────────────────────────────────────
def classify_context(
    frame: Union[np.ndarray, "PIL.Image.Image"]
) -> Tuple[Context, List[str], int]:
    """
    Parameters
    ----------
    frame : np.ndarray | PIL.Image
        Screenshot of the whole game window (1280×828).

    Returns
    -------
    context : Context
        IN_GAME / ONE_REWARD / TWO_REWARDS
    rewards : list[str]
        • []  if IN_GAME  
        • [<reward>]            if ONE_REWARD  
        • [<left_reward>, <right_reward>] if TWO_REWARDS  
          (entries are canonical names when recognised, otherwise raw OCR text)
    """
    # ------------------------------------------------------------------ #
    # 0. Accept both cv2 (numpy) or Pillow frames                        #
    # ------------------------------------------------------------------ #
    if "PIL" in str(type(frame)):              # Pillow Image?
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------ #
    # 1. OCR both reward slots                                           #
    # ------------------------------------------------------------------ #
    text_left  = _ocr(_crop(frame, _LEFT))
    text_right = _ocr(_crop(frame, _RIGHT))
    passenger_cnt = ocr.passenger_cnt(frame)
    game_over = _ocr(_crop(frame, _GAME_OVER))
    # ------------------------------------------------------------------ #
    # 2. Decide which banner state we are in                             #
    # ------------------------------------------------------------------ #
    if game_over:
        return Context.GAME_OVER, [], passenger_cnt
    if not text_left and not text_right:
        return Context.IN_GAME, [], passenger_cnt

    # single long reward: “Locomotive” is split roughly as “Loco / tive”
    if (
        "loco" in _normalise(text_left)
        and "tive" in _normalise(text_right)
        and len(text_left) <= 6
        and len(text_right) <= 6
    ):
        return Context.ONE_REWARD, [_to_reward("locomotive") or "locomotive"], passenger_cnt

    # otherwise treat as two separate rewards
    rewards = [
        _to_reward(text_left)  or text_left,
        _to_reward(text_right) or text_right,
    ]
    return Context.TWO_REWARDS, rewards, passenger_cnt



# cnt = 1
#
# while True:
#
#     # region_whole = (0, 30, 638, 472)  # (x1, y1, x2, y2)
#     frame = pyautogui.screenshot()  # pyautogui.screenshot(region=(left, top, width, height))
#     # img = np.array(frame.convert("RGB"))
#     # #img   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     # print("raw screenshot shape:", img.shape, "min/max:", img.min(), img.max())
#     # cv2.imwrite("debug_full.png", img)
#     frame.save("./ContextClassifier/context.png") # expect you run the py file under outer directory
#
#     time.sleep(0.5)
#
#
#     # Use your own region
#     region_left = (402, 610, 647-402, 50) # (x1, y1, w, h)
#     #"REGION": (610, 419, 668, 637),   (y1, x1, y2, x2) from spritex copy
#     region_right = (662, 614, 899-662, 50)
#
#     # x, y, w, h = region_left
#     # i = img[y:y+h, x:x+w]
#     # cv2.imshow("hi_test", i)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     text_left = ocr.ocr_in_region(region=region_left)
#     text_right = ocr.ocr_in_region(region=region_right)
#
#     print(f"{cnt}: Left: {text_left}\nRight: {text_right}")
#     cnt += 1
#
#     # logic to determine(can use what you prefer, e.g. a dict store all reward names)
#     print(len(text_left), len(text_right))
#     print("left:")
#     for _ in text_left:
#         print(f"*{_}*")
#
#     if text_left == "" and text_right == "" :
#         print(f"{cnt}: In gameplay context")
#     elif "Loco" in text_left and "tive" in text_right and len(text_left) <= 6 and len(text_right) <= 6:
#         print(f"{cnt}: In 1-reward context")
#         print("reward: locomotive")
#     else:
#         print(f"{cnt}: In 2-reward context")
#         print(f"left reward: {text_left}, right reward: {text_right}")
