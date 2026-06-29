"""
auto_detect.py

Automatically finds a fixed-position watermark/logo region in a video by
looking for the area that stays nearly constant across many frames while
everything else (gameplay/content) changes around it.

This needs NO logo crop image - it figures out where the watermark is by
itself, which is the right approach when the watermark sits in the same
spot for the whole video (typical for screen-recorder badges, app logos,
etc.) and you don't have (or don't want to make) a separate logo image.

Core idea:
    1. Sample N frames spread across the video
    2. Compute per-pixel variance across those frames (in grayscale)
    3. Low-variance pixels = "barely changes" = likely watermark or static UI
    4. Find a compact connected region of low variance that's small relative
       to the frame (a watermark, not the whole static background)
    5. Return that region as a bounding box, usable directly for masking +
       inpainting - same as a manually-cropped logo match would be.
"""

import cv2
import numpy as np


def detect_locked_position(
    video_path,
    logo_path,
    n_samples=25,
    score_threshold=0.5,
    scales=(0.5, 0.75, 1.0, 1.25, 1.5),
    log=print,
):
    """
    For watermarks that have their OWN internal motion/animation (so the
    plain variance-based detect_static_region above won't find them) but
    still sit in a FIXED screen position the whole video - this finds that
    fixed position by template-matching a logo crop across many sampled
    frames and taking the most common (voted) position, rather than trusting
    any single frame's match (a single frame can occasionally false-match
    somewhere else if the logo region's score isn't highest in that frame).

    Returns (x, y, w, h) of the locked position, or None if no position got
    enough agreement to be trustworthy.
    """
    import cv2
    from collections import Counter

    logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise FileNotFoundError(f"Could not read logo image: {logo_path}")
    if logo.shape[-1] == 4:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2BGR)
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_indices = np.linspace(0, total_frames - 1, num=min(n_samples, total_frames), dtype=int)

    votes = []  # (top_left, size) pairs from frames that scored well
    for idx in sorted(set(sample_indices.tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = _match_logo_local(frame_gray, logo_gray, scales)
        if result is not None:
            top_left, size, score = result
            if score >= score_threshold:
                votes.append(top_left)
    cap.release()

    if not votes:
        log(f"  [locked-detect] No frame matched the logo with score >= {score_threshold}.")
        return None

    counter = Counter(votes)
    best_position, count = counter.most_common(1)[0]
    agreement = count / len(votes)
    log(f"  [locked-detect] position {best_position} won {count}/{len(votes)} votes "
        f"({agreement:.0%} agreement)")

    if agreement < 0.4:
        log("  [locked-detect] Low agreement across frames - position may be unreliable. "
            "Consider a tighter/cleaner logo crop.")

    lw, lh = logo_gray.shape[1], logo_gray.shape[0]
    return (best_position[0], best_position[1], lw, lh)


def _match_logo_local(frame_gray, logo_gray, scales):
    """Standalone copy of the template-matching routine, kept here so this
    module has no import dependency on remove_watermark_batch.py."""
    best = None
    fh, fw = frame_gray.shape[:2]
    for scale in scales:
        lw = int(logo_gray.shape[1] * scale)
        lh = int(logo_gray.shape[0] * scale)
        if lw < 8 or lh < 8 or lw > fw or lh > fh:
            continue
        resized = cv2.resize(logo_gray, (lw, lh), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best[0]:
            best = (max_val, max_loc, (lw, lh))
    if best is None:
        return None
    score, top_left, size = best
    return top_left, size, score


def detect_static_region(
    video_path,
    n_samples=40,
    variance_threshold=12.0,
    min_area_frac=0.0005,
    max_area_frac=0.06,
    border_margin_frac=0.0,
    log=print,
):
    """
    Scans `video_path` and returns the bounding box (x, y, w, h) of the most
    likely fixed-position watermark region, or None if nothing suitable is
    found.

    Args:
        n_samples: how many frames to sample across the video for the
            variance calculation. More = more reliable, slower.
        variance_threshold: pixels with grayscale variance below this
            (across the sampled frames) are considered "static". Lower =
            stricter (only near-perfectly-still pixels count).
        min_area_frac / max_area_frac: a static region must cover between
            this fraction of total frame area to be considered a watermark
            candidate. This filters out (a) tiny noise specks and (b) large
            static backgrounds/letterboxing, which aren't watermarks.
        border_margin_frac: if > 0, restricts candidates to within this
            fraction of the frame border (watermarks are almost always
            near an edge/corner). 0 = search the whole frame.

    Returns:
        (x, y, w, h) in pixel coordinates, or None.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video reports 0 frames, cannot sample: {video_path}")

    sample_indices = np.linspace(0, total_frames - 1, num=min(n_samples, total_frames), dtype=int)
    sample_indices = sorted(set(sample_indices.tolist()))

    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()

    if len(frames) < 5:
        log(f"  [auto-detect] Only got {len(frames)} usable sample frames, too few to be reliable.")
        return None

    stack = np.stack(frames, axis=0)  # (n_samples, H, W)
    variance_map = stack.var(axis=0)  # (H, W) - low value = static pixel

    static_mask = (variance_map < variance_threshold).astype(np.uint8) * 255

    if border_margin_frac > 0:
        bx = int(width * border_margin_frac)
        by = int(height * border_margin_frac)
        edge_mask = np.zeros_like(static_mask)
        edge_mask[:by, :] = 255
        edge_mask[-by:, :] = 255
        edge_mask[:, :bx] = 255
        edge_mask[:, -bx:] = 255
        static_mask = cv2.bitwise_and(static_mask, edge_mask)

    # Clean up small noise, then find connected components of static pixels
    static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(static_mask)

    frame_area = width * height
    candidates = []
    for label_id in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[label_id]
        area_frac = area / frame_area
        if min_area_frac <= area_frac <= max_area_frac:
            # prefer regions that are reasonably box-shaped (area close to w*h),
            # which filters out scattered noise that happened to be low-variance
            fill_ratio = area / max(1, w * h)
            candidates.append((area, fill_ratio, (x, y, w, h)))

    if not candidates:
        log("  [auto-detect] No static region found matching watermark-like size/shape. "
            "Try lowering --variance-threshold or widening --max-area-frac.")
        return None

    # Prefer the largest reasonably solid candidate (most likely the actual badge,
    # not a UI sliver or noise fragment)
    candidates.sort(key=lambda c: (c[1] > 0.4, c[0]), reverse=True)
    _, _, bbox = candidates[0]
    return bbox


def bbox_to_match_format(bbox):
    """Convert (x, y, w, h) into the (top_left, size) format used by process_video's masking."""
    x, y, w, h = bbox
    return (x, y), (w, h)
