import warnings
warnings.filterwarnings("ignore")

import cv2

from vision.station_detector import annotate_stations
from vision.line_detector    import annotate_lines


IMG_PATH = "GM.png"          # <— drop in any sample screenshot


def main() -> None:
    # ──────────────── station detector ─────────────────
    annot_st, _, _ = annotate_stations(IMG_PATH)
    cv2.imwrite("debug_stations.png", annot_st)

    # ──────────────── line   detector ─────────────────
    annot_ln, polys, labels = annotate_lines(IMG_PATH, show=True)
    cv2.imwrite("debug_lines.png", annot_ln)

    # ──────────────── smoke-test summary ──────────────
    print(
        f"[OK] station+line detection ran. "
        f"Lines: {len(polys)}  unique colours: {set(labels)}"
    )


if __name__ == "__main__":
    main()