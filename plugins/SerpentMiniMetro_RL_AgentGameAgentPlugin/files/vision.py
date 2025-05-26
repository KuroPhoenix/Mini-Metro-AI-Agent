"""
Geometry-based station & line detector for Mini-Metro
"""
from .actions import LineColor as Color
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional
from .actions import StationShape     # <- your existing enum
from pathlib import Path
from collections import defaultdict
# ---------- datatypes -------------------------------------------------
@dataclass
class Station:
    shape: StationShape
    center: Tuple[int, int]           # (x, y)
    radius: int                       # for clicking
    is_interchange: bool = False

LineRoute = Dict[Color, List[Tuple[int, int]]]          # ordered points

# ---------- colour masks ---------------------------------------------
HSV_RANGES = {
    # already there
    Color.RED:      ((  0,120,80), ( 10,255,255)),
    Color.YELLOW:   (( 20,100,80), ( 35,255,255)),
    Color.GREEN:    (( 45,100,80), ( 75,255,255)),
    Color.BLUE:     ((100,100,80), (130,255,255)),
    Color.PURPLE:   ((135,100,80), (160,255,255)),

    # add new ones
    Color.CYAN:     (( 85,100,80), ( 95,255,255)),   # Nanjing cyan
    Color.PEACH:    (( 15, 70,80), ( 20,200,255)),   # Guangzhou peach
    Color.BEIGE:    (( 15, 30,80), ( 25,120,255)),   # Chongqing beige
    Color.MAGENTA:  ((160,100,80), (175,255,255)),   # Guangzhou magenta / future pink
}
# Map enum → (template contour)

def _load_templates(
    folder: Path,
    *,
    loader: Callable[[Path], Any]   = lambda p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE),
    postprocess: Callable[[Any], Any] = lambda x: x
) -> Dict[str, Any]:
    """
    Load every .png in `folder` into a dict keyed by filename (without .png).
    Applies `loader` then `postprocess` to each file.
    """
    templates: Dict[str, Any] = {}
    for p in folder.glob("*.png"):
        img = loader(p)
        tpl = postprocess(img)
        templates[p.stem] = tpl
    return templates


# vision.py
HSV_LINE_S_MIN = 80          # line > river
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

def colour_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    masks={}
    for col,(lo,hi) in HSV_RANGES.items():
        lo = np.array(lo); hi=np.array(hi)
        mask=cv2.inRange(hsv,lo,hi)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)      # despeckle
        mask=cv2.erode(mask,kernel,iterations=1)               # kill river
        masks[col]=mask
    return masks

# ---------- station detector -----------------------------------------
STATION_DIR = Path(__file__).parent / "assets/templates_resized/stations_resized"
ICON_DIR  = Path(__file__).parent / "assets/templates_resized"
DIGIT_DIR = ICON_DIR / "digits_resized"
POPUP_DIR = Path(__file__).parent / "assets/templates_resized"
def _contour_from_img(img):
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    cnts, _   = cv2.findContours(thresh,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea)

# populate the station‐shape templates
# station‐shape contours
_template_contours = _load_templates(
    STATION_DIR,
    postprocess=_contour_from_img
)

# raw icons for matchTemplate
TEMPLATES = _load_templates(
    ICON_DIR,
    postprocess=lambda img: img if img is not None else np.zeros((1,1), np.uint8)
)

# digit images
DIGIT_T = _load_templates(
    DIGIT_DIR,
    postprocess=lambda img: img if img is not None else np.zeros((1,1), np.uint8)
)

# popup icons
_POPUP_TEMPLATES = _load_templates(
    POPUP_DIR,
    postprocess=lambda img: img if img is not None else np.zeros((1,1), np.uint8)
)


def detect_stations(bgr: np.ndarray,
                    debug_img: Optional[np.ndarray] = None
                   ) -> List[Station]:
    # 1) build one white‐only mask in HSV:
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv,
                        np.array([  0,   0,200]),   # H any, S ≤  30, V ≥ 200
                        np.array([180,  30,255]))   # capture all bright whites

    # 2) open then close to dump tiny specks (passengers)
    kern_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN,  kern_small, iterations=2)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kern_small, iterations=1)

    # 3) carve out *any* coloured-line pixels so that stations on tracks
    #    stay islands of pure white
    line_mask = np.zeros_like(white)
    for lo,hi in HSV_RANGES.values():
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)     # despeckle
        m = cv2.dilate(   m, kernel, iterations=2)          # eat thin “river” strokes
        line_mask |= m
    white[line_mask>0] = 0

    # 4) find contours *once* in that clean white mask
    cnts,_ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H,W      = white.shape
    min_area = 150         # toss anything < 150px²
    max_area = 0.5*H*W     # toss anything huge

    out: List[Station] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue

        peri   = cv2.arcLength(c, True)
        eps    = 0.02 * peri
        circ   = 4*np.pi*area/(peri*peri)
        approx = cv2.approxPolyDP(c, eps, True)
        v      = len(approx)

        # quick aspect‐ratio guard
        x,y,wc,hc = cv2.boundingRect(c)
        a = max(wc/hc, hc/wc)
        if a>8 or (a>3 and circ<0.5):
            continue

        # (optional) draw the poly you’re testing
        if debug_img is not None:
            cv2.drawContours(debug_img, [approx], -1, (255,0,0), 1)

        # classify shape by circularity & # vertices
        shape = None
        if   circ>0.85:                  shape = StationShape.CIRCLE
        elif v==3:                       shape = StationShape.TRIANGLE
        elif v==4:
            _,(rw,rh),ang = cv2.minAreaRect(c)
            if abs(rw-rh)<0.1*rw:
                shape = StationShape.GEM if abs((ang%90)-45)<10 \
                        else StationShape.SQUARE
        elif v==5:
            shape = (StationShape.PENTAGON
                     if cv2.isContourConvex(approx)
                     else StationShape.DIAMOND)

        # fallback: template match *only* if very tight
        if shape is None:
            best_score, best_key = float("inf"), None
            for key, tpl in _template_contours.items():
                score = cv2.matchShapes(c, tpl,
                                         cv2.CONTOURS_MATCH_I1, 0.0)
                if score<best_score:
                    best_score,best_key = score,key
            if best_score<0.15:
                shape = StationShape[best_key.upper()]

        if shape is None:
            continue     # give up on weird junk

        # toss any tiny passenger‐circle that slipped through
        (cx,cy),r = cv2.minEnclosingCircle(c)
        if r<8:
            continue

        is_intchg  = _is_interchange(c)
        out.append(Station(shape, (int(cx),int(cy)), int(r), is_intchg))

    return out


def _is_interchange(c):
    hull   = cv2.convexHull(c)
    ratio  = cv2.contourArea(hull) / cv2.contourArea(c)
    return ratio > 1.20

def match_via_templates(cnt):
    """
    Return the StationShape whose template contour
    best matches the given contour `cnt`.
    """
    best_shape = None
    best_score = float("inf")

    for shape, tmpl_cnt in _template_contours.items():
        # I1 is usually fine; you can try I2 or I3 if you like
        score = cv2.matchShapes(cnt, tmpl_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
        if score < best_score:
            best_score, best_shape = score, shape

    return best_shape

# ---------- line extractor -------------------------------------------
def detect_lines(bgr: np.ndarray) -> LineRoute:
    masks = colour_masks(bgr)
    routes: LineRoute = defaultdict(list)
    for color, mask in masks.items():
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if cv2.arcLength(c, False) < 100: continue   # ignore specks
            poly = cv2.approxPolyDP(c, x3, False).squeeze()
            routes[color].append([(int(x),int(y)) for x,y in poly])
    return routes


def match_icon(gray, tmpl, thr=0.8):
    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    _, m, _, _ = cv2.minMaxLoc(res)
    return m >= thr


def _best_match(gray_roi, tmpl, thr=0.8):
    res = cv2.matchTemplate(gray_roi, tmpl, cv2.TM_CCOEFF_NORMED)
    _,val,_,loc = cv2.minMaxLoc(res)
    return (val, loc) if val>=thr else (None, None)

def read_digit(gray_roi):
    best = None
    for d,tmpl in DIGIT_T.items():
        v,_ = _best_match(gray_roi, tmpl, 0.7)
        if v and (best is None or v>best[0]):
            best = (v, d)
    return int(best[1]) if best else 0

def detect_inventory(full_bgr) -> Dict[str,int]:
    h,w,_ = full_bgr.shape
    roi   = full_bgr[h-90:h, 0:w]                 # bottom strip
    g     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    store = {}
    x = 0
    for name,tmpl in TEMPLATES.items():
        val,loc = _best_match(g, tmpl)
        if val:
            x0,y0 = loc
            # assume digit is 20 px right of icon
            digit_roi = g[y0:y0+tmpl.shape[0], x0+tmpl.shape[1]+4:x0+tmpl.shape[1]+20]
            store[name] = read_digit(digit_roi)
            x = x0+tmpl.shape[1]+24
    return store        # e.g. {"loco":2,"carriage":1,"tunnel":0}

# ---------- weekly upgrade popup -------------------------------------
# works for 1280-wide window.  Tune ROIs once if you play larger.

def _popup_roi(bgr):
    """return the central 600×350 region where the two icons live"""
    h,w,_ = bgr.shape
    return bgr[h//2-175:h//2+175, w//2-300:w//2+300]

def detect_weekly_offer(full_bgr) -> List[str]:
    """
    Return a list with *two* strings (icon names) if the Monday popup
    is visible, else empty list.
    """
    roi   = _popup_roi(full_bgr)
    g     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    offers=[]
    for name,tmpl in _POPUP_TEMPLATES.items():
        if tmpl is None:           # template missing
            continue
        res = cv2.matchTemplate(g, tmpl, cv2.TM_CCOEFF_NORMED)
        _,val,_,loc = cv2.minMaxLoc(res)
        if val>0.85:
            offers.append(name)
    return offers[:2]              # at most two

