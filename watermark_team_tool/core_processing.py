#!/usr/bin/env python3
"""
remove_watermark_batch.py

Removes a moving (but visually constant) logo/watermark from one or many
videos, where EACH video can have its OWN logo.

Pairing convention:
    For a video named "myvideo.mp4", the script looks for a matching logo at
    "myvideo_logo.png" in the same folder. Any video without a matching logo
    file is skipped (reported, not silently ignored).

Folder layout example (e.g. a Google Drive folder):
    video1.mp4
    video1_logo.png
    video2.mp4
    video2_logo.png
    ...

CLI usage (run from a terminal, not inside a notebook cell):
    python3 remove_watermark_batch.py --folder /path/to/folder --output-dir /path/to/output
    python3 remove_watermark_batch.py --folder /path/to/folder --output-dir /path/to/output --preview

Colab / notebook usage:
    Don't run this file's main() inside a notebook cell (argparse will fail
    on Colab's own launcher args). Instead import and call the batch function
    directly - see the companion notebook cells / README for the exact code.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from auto_detect import detect_static_region, bbox_to_match_format, detect_locked_position

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Core single-video logic (same algorithm as the single-video script)
# ---------------------------------------------------------------------------

def load_logo(logo_path: str):
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise FileNotFoundError(f"Could not read logo image: {logo_path}")
    if logo.shape[-1] == 4:
        logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2BGR)
    return logo


def match_logo(frame_gray, logo_gray, scales):
    best = None  # (score, top_left, (w, h))
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


def build_mask(frame_shape, top_left, size, padding):
    h, w = frame_shape[:2]
    x, y = top_left
    lw, lh = size
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w, x + lw + padding)
    y1 = min(h, y + lh + padding)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask, (x0, y0, x1, y1)


def mux_audio(original_path, silent_video_path, output_path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(original_path)],
        capture_output=True, text=True,
    )
    has_audio = probe.stdout.strip() != ""

    if not has_audio:
        Path(silent_video_path).rename(output_path)
        return "no_audio_in_source"

    cmd = [
    "ffmpeg", "-y",
    "-i", str(silent_video_path),
    "-i", str(original_path),

    "-map", "0:v:0",
    "-map", "1:a:0?",

    "-c:v", "libx264",
    "-preset", "slow",
    "-crf", "18",
    "-pix_fmt", "yuv420p",

    "-c:a", "aac",
    "-b:a", "192k",

    "-movflags", "+faststart",
    "-shortest",

    str(output_path),
]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"audio_mux_failed: {result.stderr[-500:]}"

    Path(silent_video_path).unlink(missing_ok=True)
    return "ok"


def process_video(
    input_path,
    logo_path=None,
    output_path=None,
    preview=False,
    padding=6,
    threshold=0.55,
    scales=(0.5, 0.75, 1.0, 1.25, 1.5),
    sample_every=5,
    inpaint_radius=4,
    inpaint_method="telea",
    progress_every=60,
    auto_detect=False,
    auto_detect_samples=40,
    auto_detect_variance_threshold=12.0,
    locked_position=False,
    locked_position_samples=25,
    fully_auto=False,
    fully_auto_padding=14,
    log=print,
):
    """
    Processes a single video. Returns a small dict report so batch_process
    can summarize results across many videos.

    Modes:
      - logo_path given (default): per-frame template matching against that
        logo image. Handles a watermark that moves around within the video.
      - auto_detect=True (no logo_path needed): detects a fixed-position
        static region via temporal variance. Use this when the watermark
        sits in one spot and has no internal motion/animation of its own.
      - locked_position=True (logo_path required): template-matches the logo
        across several sample frames, takes the most-voted position, then
        uses that ONE fixed position for the whole video.
      - fully_auto=True (no logo_path needed, no mode selection needed):
        tries static-variance detection first, falls back to a bootstrapped
        template search (guesses corner/edge candidate regions and checks
        which one matches consistently across frames) if that fails. This
        is the mode the web tool uses - zero manual input required.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log(f"  [info] {Path(input_path).name}: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    fixed_bbox = None
    logo = None
    logo_gray = None
    detect_method = None
    detect_confidence = None

    if fully_auto:
        from auto_locked_detect import detect_watermark_no_crop
        cap.release()
        fixed_bbox, detect_method, detect_confidence = detect_watermark_no_crop(
            input_path, log=log,
        )
        if fixed_bbox is None:
            raise RuntimeError(
                "Could not automatically find a watermark region in this video. "
                "The watermark may not be in a fixed position, or may be too subtle "
                "to detect reliably - manual logo crop mode would be needed instead."
            )
        log(f"  [fully-auto] watermark region: {fixed_bbox} via {detect_method} "
            f"({detect_confidence:.0%} confidence)")
        padding = fully_auto_padding  # auto-detected boxes can undershoot slightly; pad generously
        cap = cv2.VideoCapture(str(input_path))
    elif locked_position:
        if logo_path is None:
            raise ValueError("locked_position=True requires logo_path (a crop of the logo).")
        cap.release()
        fixed_bbox = detect_locked_position(
            input_path, logo_path, n_samples=locked_position_samples, log=log,
        )
        if fixed_bbox is None:
            raise RuntimeError(
                "Could not lock a reliable position for this logo. Try a tighter/cleaner "
                "logo crop, or fall back to per-frame matching (locked_position=False)."
            )
        log(f"  [locked-position] watermark region: {fixed_bbox} (x, y, w, h)")
        cap = cv2.VideoCapture(str(input_path))
    elif auto_detect:
        cap.release()  # detect_static_region opens its own capture
        fixed_bbox = detect_static_region(
            input_path,
            n_samples=auto_detect_samples,
            variance_threshold=auto_detect_variance_threshold,
            log=log,
        )
        if fixed_bbox is None:
            raise RuntimeError(
                "Auto-detect found no watermark-like static region in this video. "
                "Try lowering --auto-detect-variance-threshold, or use a manual logo crop instead."
            )
        log(f"  [auto-detect] watermark region: {fixed_bbox} (x, y, w, h)")
        cap = cv2.VideoCapture(str(input_path))
    else:
        if logo_path is None:
            raise ValueError("Either logo_path or auto_detect=True must be provided.")
        logo = load_logo(str(logo_path))
        logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

    # Write to a temp silent video first; audio gets muxed in afterward.
    tmp_video = str(Path(output_path).with_suffix("")) + "_silent_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

    inpaint_flag = cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS

    last_match = bbox_to_match_format(fixed_bbox) if fixed_bbox is not None else None
    frame_idx = 0
    matches_found = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not auto_detect and not locked_position and not fully_auto:
            do_detect = (frame_idx % sample_every == 0) or (last_match is None)
            if do_detect:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = match_logo(frame_gray, logo_gray, scales)
                if result is not None:
                    top_left, size, score = result
                    if score >= threshold:
                        last_match = (top_left, size)
                        matches_found += 1
        else:
            matches_found += 1  # fixed region applies to every frame by definition

        if preview:
            out_frame = frame.copy()
            if last_match is not None:
                (x, y), (lw, lh) = last_match
                cv2.rectangle(out_frame, (x, y), (x + lw, y + lh), (0, 0, 255), 2)
        else:
            if last_match is not None:
                mask, _ = build_mask(frame.shape, last_match[0], last_match[1], padding)
                out_frame = cv2.inpaint(frame, mask, inpaint_radius, inpaint_flag)
            else:
                out_frame = frame

        writer.write(out_frame)
        frame_idx += 1
        if progress_every and frame_idx % progress_every == 0:
            log(f"    ...{frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    sampled = max(1, frame_idx // sample_every + 1) if (not auto_detect and not locked_position and not fully_auto) else max(1, frame_idx)
    detect_rate = matches_found / sampled

    report = {
        "video": str(input_path),
        "logo": str(logo_path) if logo_path else "auto-detect",
        "bbox": fixed_bbox,
        "output": str(output_path),
        "frames": frame_idx,
        "detect_rate": detect_rate,
        "detect_method": detect_method,
        "detect_confidence": detect_confidence,
        "status": "ok",
    }

    if preview:
        Path(tmp_video).rename(output_path)
        report["status"] = "preview_done"
        return report

    audio_status = mux_audio(input_path, tmp_video, output_path)
    report["audio"] = audio_status
    if audio_status not in ("ok", "no_audio_in_source"):
        report["status"] = "audio_failed"

    if detect_rate < 0.3:
        report["status"] = "low_detection_warning"

    return report


# ---------------------------------------------------------------------------
# Batch logic
# ---------------------------------------------------------------------------

def find_pairs(folder, logo_suffix="_logo"):
    """
    Scans `folder` for video files and matching "<stem>_logo.<ext>" images.
    Returns (pairs, unpaired_videos) where pairs is a list of (video_path, logo_path).
    """
    folder = Path(folder)
    videos = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    pairs = []
    unpaired = []
    for video in videos:
        logo_candidates = list(folder.glob(f"{video.stem}{logo_suffix}.*"))
        # prefer common image extensions if multiple matches somehow exist
        logo_candidates = [c for c in logo_candidates if c.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if logo_candidates:
            pairs.append((video, logo_candidates[0]))
        else:
            unpaired.append(video)

    return pairs, unpaired


def batch_process(
    folder,
    output_dir,
    preview=False,
    logo_suffix="_logo",
    output_suffix="_clean",
    auto_detect=False,
    auto_detect_samples=40,
    auto_detect_variance_threshold=12.0,
    locked_position=False,
    locked_position_samples=25,
    log=print,
    **process_kwargs,
):
    """
    Processes every video in `folder`, writing results to `output_dir`.
    Returns a list of per-video report dicts.

    Three modes:
      - auto_detect=False, locked_position=False (default): looks for a
        matching "<name>_logo.png" per video, template-matches it on every
        sampled frame. Use when the watermark moves around within a video.
      - auto_detect=True: no logo files needed at all - each video's
        watermark region is detected via temporal variance. Use when the
        watermark sits in a fixed spot and has no animation of its own.
      - locked_position=True: needs "<name>_logo.png" per video (same
        pairing as default mode), but votes across sample frames to lock
        one fixed position for the whole video instead of re-matching every
        frame. Use when the watermark sits in a fixed spot but ALSO has its
        own internal animation (so auto_detect's variance check won't find
        it), e.g. an animated ad/promo badge.

    Extra keyword args (threshold, scales, sample_every, padding, etc.) are
    passed straight through to process_video for every video in the batch.
    """
    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    needs_logo_files = not auto_detect  # both default mode and locked_position need logo crops

    if not needs_logo_files:
        videos = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        pairs = [(v, None) for v in videos]
        unpaired = []
        log(f"[batch] Auto-detect mode: found {len(videos)} video(s) in {folder}, "
            f"no logo files needed")
    else:
        pairs, unpaired = find_pairs(folder, logo_suffix=logo_suffix)
        log(f"[batch] Found {len(pairs)} video/logo pair(s) in {folder}")
        if unpaired:
            log(f"[batch] WARNING: {len(unpaired)} video(s) skipped, no matching "
                f"'<name>{logo_suffix}.png' found: {[p.name for p in unpaired]}")

    reports = []
    for i, (video_path, logo_path) in enumerate(pairs, start=1):
        suffix = "_preview" if preview else output_suffix
        out_name = f"{video_path.stem}{suffix}{video_path.suffix}"
        out_path = output_dir / out_name

        if not needs_logo_files:
            log(f"[batch] ({i}/{len(pairs)}) {video_path.name}  <-  auto-detecting watermark")
        elif locked_position:
            log(f"[batch] ({i}/{len(pairs)}) {video_path.name}  <-  logo: {logo_path.name} (locked position)")
        else:
            log(f"[batch] ({i}/{len(pairs)}) {video_path.name}  <-  logo: {logo_path.name}")

        try:
            report = process_video(
                input_path=video_path,
                logo_path=logo_path,
                output_path=out_path,
                preview=preview,
                auto_detect=auto_detect,
                auto_detect_samples=auto_detect_samples,
                auto_detect_variance_threshold=auto_detect_variance_threshold,
                locked_position=locked_position,
                locked_position_samples=locked_position_samples,
                log=log,
                **process_kwargs,
            )
        except Exception as e:
            report = {
                "video": str(video_path),
                "logo": str(logo_path) if logo_path else "auto-detect",
                "output": str(out_path),
                "status": "error",
                "error": str(e),
            }
            log(f"  [error] {video_path.name}: {e}")

        reports.append(report)

    # Summary
    log("\n[batch] ==== Summary ====")
    for r in reports:
        name = Path(r["video"]).name
        status = r.get("status", "unknown")
        if status in ("ok", "preview_done"):
            rate = r.get("detect_rate")
            rate_str = f", detect rate {rate:.0%}" if rate is not None else ""
            log(f"  OK      {name}{rate_str} -> {Path(r['output']).name}")
        elif status == "low_detection_warning":
            log(f"  WARNING {name}: low detection rate ({r.get('detect_rate', 0):.0%}) "
                f"- check {Path(r['output']).name} and consider adjusting threshold/scales for this video")
        else:
            log(f"  FAILED  {name}: {r.get('error', r.get('audio', status))}")
    for video_path in unpaired:
        log(f"  SKIPPED {video_path.name}: no matching logo file found")

    return reports


# ---------------------------------------------------------------------------
# CLI entry point (terminal use only - see module docstring for Colab usage)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-remove per-video logos/watermarks from all videos in a folder."
    )
    parser.add_argument("--folder", required=True,
                         help="Folder containing videos and their matching '<name>_logo.png' files")
    parser.add_argument("--output-dir", required=True, help="Folder to write cleaned videos to")
    parser.add_argument("--preview", action="store_true",
                         help="Draw detection boxes instead of inpainting, to sanity check first")
    parser.add_argument("--logo-suffix", default="_logo",
                         help="Suffix used to find each video's logo file (default: _logo)")
    parser.add_argument("--auto-detect", action="store_true",
                         help="Detect each video's watermark automatically; no logo files needed. "
                              "Use this when the watermark sits in a fixed spot per video.")
    parser.add_argument("--auto-detect-samples", type=int, default=40,
                         help="Frames sampled to detect the static watermark region (default: 40)")
    parser.add_argument("--auto-detect-variance-threshold", type=float, default=12.0,
                         help="Lower = stricter static-pixel detection (default: 12.0)")
    parser.add_argument("--padding", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--scales", default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--inpaint-radius", type=int, default=4)
    parser.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea")
    args = parser.parse_args()

    scales = tuple(float(s) for s in args.scales.split(","))

    batch_process(
        folder=args.folder,
        output_dir=args.output_dir,
        preview=args.preview,
        logo_suffix=args.logo_suffix,
        auto_detect=args.auto_detect,
        auto_detect_samples=args.auto_detect_samples,
        auto_detect_variance_threshold=args.auto_detect_variance_threshold,
        padding=args.padding,
        threshold=args.threshold,
        scales=scales,
        sample_every=args.sample_every,
        inpaint_radius=args.inpaint_radius,
        inpaint_method=args.inpaint_method,
    )


if __name__ == "__main__":
    main()
