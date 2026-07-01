"""
auto_locked_detect.py

Detects a fixed-position watermark WITHOUT needing a pre-made logo crop
image, even when the watermark has its own internal animation (so plain
variance-based detection in auto_detect.py won't find it).

How it works:
    1. Try the cheap variance-based static-region detector first (handles
       the common case: a non-animated badge/logo).
    2. If that fails, fall back to a self-bootstrapping approach:
       - Take one frame as a reference.
       - Scan candidate regions along the frame's border/corners (where
         watermarks almost always live).
       - For each candidate region, crop it out of the reference frame and
         use THAT crop as a template to match against several other sample
         frames - if the region really is a fixed watermark, it should
         match at the SAME position in most of those frames even though the
         watermark itself might shimmer/animate slightly.
       - Score each candidate by (a) how many frames it matched in, at (b)
         the same position, and pick the best one.
    This needs no manual cropping step - it bootstraps its own template
    from a guessed region and checks whether that guess holds up.
"""

import cv2
import numpy as np
from collections import Counter

from auto_detect import detect_static_region


def _grid_candidates(
    width,
    height,
    box_w_fracs=(0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30),
    box_h_fracs=(0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22),
):
    """
    Generate candidate watermark regions around the screen edges.
    """

    candidates = []

    for wf in box_w_fracs:
        for hf in box_h_fracs:

            w = int(width * wf)
            h = int(height * hf)

            margin = int(min(width, height) * 0.01)

            positions = {
                "top_left": (margin, margin),

                "top_center": (
                    (width - w) // 2,
                    margin,
                ),

                "top_right": (
                    width - w - margin,
                    margin,
                ),

                "middle_left": (
                    margin,
                    (height - h) // 2,
                ),

                "middle_right": (
                    width - w - margin,
                    (height - h) // 2,
                ),

                "bottom_left": (
                    margin,
                    height - h - margin,
                ),

                "bottom_center": (
                    (width - w) // 2,
                    height - h - margin,
                ),

                "bottom_right": (
                    width - w - margin,
                    height - h - margin,
                ),
            }

            for x, y in positions.values():
                if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                    candidates.append((x, y, w, h))

    return candidates


def _match_score(frame_gray, template_gray):

    methods = [
        cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR_NORMED,
    ]

    best_score = -1
    best_loc = (0, 0)

    for method in methods:
        result = cv2.matchTemplate(frame_gray, template_gray, method)
        _, score, _, loc = cv2.minMaxLoc(result)

        if score > best_score:
            best_score = score
            best_loc = loc

    return best_score, best_loc


def detect_watermark_no_crop(
    video_path,
    n_grid_samples=15,
    n_verify_samples=40,
    match_threshold=0.35,
    agreement_threshold=0.35,
    log=print,
):
    """
    Attempts to find a fixed-position watermark region with no logo image
    supplied. Returns (bbox, method, confidence) where method is one of
    "static_variance" or "bootstrapped_template", or (None, None, 0) if
    nothing confident was found.
    """
    # --- Attempt 1: cheap variance-based detection (handles non-animated logos) ---
    bbox = detect_static_region(video_path, log=lambda *a, **k: None)  # quiet on first try
    if bbox is not None:
        log(f"  [no-crop-detect] Found via static-variance method: {bbox}")
        return bbox, "static_variance", 1.0

    log("  [no-crop-detect] Static-variance method found nothing; trying bootstrapped template search...")

    # --- Attempt 2: bootstrap a template from corner-region candidates ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames < 10:
        cap.release()
        return None, None, 0

    # Reference frame: pick one from the middle (avoids title-card intros/outros)
    ref_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ret, ref_frame = cap.read()
    if not ret:
        cap.release()
        return None, None, 0
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    # Sample several OTHER frames spread across the video to verify candidates against
    verify_indices = np.linspace(0, total_frames - 1, num=min(n_verify_samples, total_frames), dtype=int)
    verify_frames_gray = []
    for idx in sorted(set(verify_indices.tolist())):
        if idx == ref_idx:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, f = cap.read()
        if ret:
            verify_frames_gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(verify_frames_gray) < 5:
        log("  [no-crop-detect] Not enough usable frames to verify candidates.")
        return None, None, 0

    candidates = _grid_candidates(width, height)
    log(f"  [no-crop-detect] Testing {len(candidates)} candidate regions against "
        f"{len(verify_frames_gray)} frames...")

    best = None  # (agreement_score, bbox)
    for (x, y, w, h) in candidates:
        template = ref_gray[y:y + h, x:x + w]
        if template.size == 0:
            continue

        matched_positions = []
        for vframe in verify_frames_gray:
            score, loc = _match_score(vframe, template)
            if score >= match_threshold:
                matched_positions.append(loc)

        if not matched_positions:
            continue

        # The candidate is good if it consistently matches near its OWN
        # position across frames (proving it's a fixed feature, not a
        # coincidental texture match that wanders around)
        counter = Counter(matched_positions)
        _, top_count = counter.most_common(1)[0]
        agreement = top_count / len(verify_frames_gray)

        if best is None or agreement > best[0]:
            best = (agreement, (x, y, w, h))

    if best is None or best[0] < agreement_threshold:
        found_agreement = best[0] if best else 0
        log(f"  [no-crop-detect] Best candidate only reached {found_agreement:.0%} agreement "
            f"(need >= {agreement_threshold:.0%}). No confident watermark region found.")
        return None, None, (best[0] if best else 0)

    agreement, bbox = best
    log(f"  [no-crop-detect] Found via bootstrapped template: {bbox}, {agreement:.0%} agreement")
    return bbox, "bootstrapped_template", agreement
