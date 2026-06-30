import cv2
import numpy as np
print("LOADED:", __file__)
def detect_best_region(frame):
    
    print("========== USING LOGO DETECTOR V2 ==========")

    h, w = frame.shape[:2]

    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best = None

    widths = [0.08, 0.10, 0.12, 0.15, 0.18, 0.22]
    heights = [0.06, 0.08, 0.10, 0.12, 0.15]

    margin = int(min(w, h) * 0.01)

    for wf in widths:
        for hf in heights:

            bw = int(w * wf)
            bh = int(h * hf)

            positions = [

                ("top_left", margin, margin),

                ("top_center", (w - bw) // 2, margin),

                ("top_right", w - bw - margin, margin),

                ("bottom_left", margin, h - bh - margin),

                ("bottom_center", (w - bw) // 2, h - bh - margin),

                ("bottom_right", w - bw - margin, h - bh - margin),

                ("left_middle", margin, (h - bh) // 2),

                ("right_middle", w - bw - margin, (h - bh) // 2),
            ]

            for name, x, y in positions:

                roi = gray[y:y+bh, x:x+bw]

                if roi.size == 0:
                    continue

                edges = cv2.Canny(roi, 60, 150)

                edge_score = np.sum(edges)

                variance = np.var(roi)

                score = edge_score + variance * 5

                if best is None or score > best[0]:
                    best = (score, name, x, y, bw, bh)

    score, name, x, y, bw, bh = best
    

    # Expand the detected region to cover the full logo.
    expand_x = int(bw * 0.75)
    expand_y = int(bh * 0.50)

    x = max(0, x - expand_x)
    y = max(0, y - expand_y)

    bw = min(w - x, bw + expand_x * 2)
    bh = min(h - y, bh + expand_y * 2)

    return score, name, x, y, bw, bh