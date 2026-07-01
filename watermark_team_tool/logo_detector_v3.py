import importlib
import numpy as np
from collections import Counter

cv2 = importlib.import_module("cv2")

def boxes_are_close(box1, box2, tolerance=25):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    return (
        abs(x1 - x2) <= tolerance and
        abs(y1 - y2) <= tolerance and
        abs(w1 - w2) <= tolerance and
        abs(h1 - h2) <= tolerance
    )

from logo_detector_v2 import detect_best_region


def detect_best_region_v3(video_path, samples=10):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        min(samples, total_frames),
        dtype=int,
    )

    detections = []

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        ret, frame = cap.read()

        if not ret:
            continue

        score, name, x, y, w, h = detect_best_region(frame)

        detections.append(
            (
                round(x / 5) * 5,
                round(y / 5) * 5,
                round(w / 5) * 5,
                round(h / 5) * 5,
                score,
            )
        )

    cap.release()

    if not detections:
        return None

    clusters = []

    for x, y, w, h, score in detections:

        found = False

        for cluster in clusters:

            if boxes_are_close(
                (x, y, w, h),
                cluster["box"]
            ):

                cluster["boxes"].append((x, y, w, h))
                cluster["scores"].append(score)

                found = True
                break

        if not found:

            clusters.append({

                "box": (x, y, w, h),

                "boxes": [(x, y, w, h)],

                "scores": [score],
            })

    best_cluster = max(
        clusters,
        key=lambda c: len(c["boxes"])
    )

    xs = [b[0] for b in best_cluster["boxes"]]
    ys = [b[1] for b in best_cluster["boxes"]]
    ws = [b[2] for b in best_cluster["boxes"]]
    hs = [b[3] for b in best_cluster["boxes"]]

    x = int(np.mean(xs))
    y = int(np.mean(ys))
    w = int(np.mean(ws))
    h = int(np.mean(hs))

    score = float(np.mean(best_cluster["scores"]))

    confidence = (
        len(best_cluster["boxes"])
        / len(detections)
    )

    return (
        score,
        confidence,
        x,
        y,
        w,
        h,
    )