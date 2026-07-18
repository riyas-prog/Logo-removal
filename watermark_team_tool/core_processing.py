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
import time
from pathlib import Path
from background_builder import BackgroundBuilder

import cv2
import numpy as np
from smart_mask import refine_mask
from comparison_video import create_side_by_side

from auto_detect import detect_static_region, bbox_to_match_format, detect_locked_position
from logo_detector_v2 import detect_best_region
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
USE_BACKGROUND_BUILDER = False


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
    """
    Improved logo matcher:
    - Searches many scales
    - Uses grayscale matching
    - Uses edge matching
    - Keeps whichever score is strongest
    """

    best = None
    fh, fw = frame_gray.shape[:2]

    # Edge version of frame
    frame_edge = cv2.Canny(frame_gray, 80, 160)

    for scale in scales:

        lw = int(logo_gray.shape[1] * scale)
        lh = int(logo_gray.shape[0] * scale)

        if lw < 8 or lh < 8:
            continue

        if lw > fw or lh > fh:
            continue

        resized = cv2.resize(
            logo_gray,
            (lw, lh),
            interpolation=cv2.INTER_AREA,
        )

        # --------------------------
        # Grayscale template match
        # --------------------------
        result_gray = cv2.matchTemplate(
            frame_gray,
            resized,
            cv2.TM_CCOEFF_NORMED,
        )

        _, gray_score, _, gray_loc = cv2.minMaxLoc(result_gray)

        # --------------------------
        # Edge template match
        # --------------------------
        logo_edge = cv2.Canny(resized, 80, 160)

        result_edge = cv2.matchTemplate(
            frame_edge,
            logo_edge,
            cv2.TM_CCOEFF_NORMED,
        )

        _, edge_score, _, edge_loc = cv2.minMaxLoc(result_edge)

        # Keep the stronger result
        if edge_score > gray_score:
            score = edge_score
            loc = edge_loc
        else:
            score = gray_score
            loc = gray_loc

        if best is None or score > best[0]:
            best = (score, loc, (lw, lh))

    if best is None:
        return None

    score, top_left, size = best
    return top_left, size, score

def build_mask(frame_shape, top_left, size, padding):
    h, w = frame_shape[:2]

    x, y = top_left
    lw, lh = size

    # Slightly expand the selected region
    expand = 3

    x0 = max(0, x - padding - expand)
    y0 = max(0, y - padding - expand)
    x1 = min(w, x + lw + padding + expand)
    y1 = min(h, y + lh + padding + expand)

    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a filled rectangle
    cv2.rectangle(
        mask,
        (x0, y0),
        (x1, y1),
        255,
        -1
    )

    # Smooth rectangle corners
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Soft edges
    mask = cv2.GaussianBlur(
        mask,
        (15, 15),
        0
    )

    return mask, (x0, y0, x1, y1)


def mux_audio(original_path, silent_video_path, output_path):

    original_path = Path(original_path)
    silent_video_path = Path(silent_video_path)
    output_path = Path(output_path)

    # --------------------------------------------------
    # REMOVE ANY OLD / PARTIAL FINAL OUTPUT
    # --------------------------------------------------

    output_path.unlink(
        missing_ok=True
    )

    # --------------------------------------------------
    # VERIFY TEMP VIDEO EXISTS
    # --------------------------------------------------

    if not silent_video_path.exists():

        raise RuntimeError(
            f"Silent temporary video was not created: "
            f"{silent_video_path}"
        )

    if silent_video_path.stat().st_size < 1000:

        raise RuntimeError(
            f"Silent temporary video is too small: "
            f"{silent_video_path.stat().st_size} bytes"
        )

    # --------------------------------------------------
    # CHECK WHETHER ORIGINAL VIDEO HAS AUDIO
    # --------------------------------------------------

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(original_path),
        ],
        capture_output=True,
        text=True,
    )

    has_audio = (
        probe.returncode == 0 and
        probe.stdout.strip() != ""
    )

    # --------------------------------------------------
    # BUILD FFMPEG COMMAND
    # --------------------------------------------------

    cmd = [
        "ffmpeg",
        "-y",

        "-i",
        str(silent_video_path),
    ]

    if has_audio:

        cmd += [
            "-i",
            str(original_path),

            "-map",
            "0:v:0",

            "-map",
            "1:a:0?",
        ]

    else:

        cmd += [
            "-map",
            "0:v:0",
        ]

    # --------------------------------------------------
    # WEB / WINDOWS / BROWSER COMPATIBLE VIDEO
    # --------------------------------------------------

    cmd += [
    "-c:v",
    "libx264",

    "-preset",
    "fast",

    "-crf",
    "18",

    "-pix_fmt",
    "yuv420p",
]

    if has_audio:

        cmd += [
            "-c:a",
            "aac",

            "-b:a",
            "192k",
        ]

    cmd += [
        "-movflags",
        "+faststart",
    ]

    if has_audio:

        cmd += [
            "-shortest",
        ]

    cmd += [
        str(output_path),
    ]

    # --------------------------------------------------
    # RUN FFMPEG
    # --------------------------------------------------

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    # --------------------------------------------------
    # FFMPEG FAILURE
    # --------------------------------------------------

    if result.returncode != 0:

        output_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"""
            FFmpeg failed

            Return code: {result.returncode}

            STDOUT: {result.stdout}

            STDERR: {result.stderr}
            """
        )

    # --------------------------------------------------
    # VERIFY OUTPUT EXISTS AND HAS REAL DATA
    # --------------------------------------------------

    if not output_path.exists():

        raise RuntimeError(
            "FFmpeg completed but final output "
            "file was not created."
        )

    output_size = output_path.stat().st_size

    if output_size < 1000:

        output_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            f"Final MP4 is too small: "
            f"{output_size} bytes"
        )

    # --------------------------------------------------
    # VALIDATE FINAL MP4 WITH FFPROBE
    # --------------------------------------------------

    validation = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",

            "-select_streams",
            "v:0",

            "-show_entries",
            "stream=codec_name,duration",

            "-of",
            "default=noprint_wrappers=1",

            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    if (
        validation.returncode != 0 or
        not validation.stdout.strip()
    ):

        validation_error = (
            validation.stderr.strip()
            or
            "ffprobe could not find a valid video stream"
        )

        output_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Final MP4 validation failed: "
            + validation_error
        )

    # --------------------------------------------------
    # SUCCESS — DELETE TEMP VIDEO
    # --------------------------------------------------

    silent_video_path.unlink(
        missing_ok=True
    )

    if has_audio:
        return "ok"

    return "no_audio_in_source"


def adaptive_remove_region(
    frame,
    mask,
    bbox,
    feather=15,
    background_roi=None,
):
    """
    Remove a selected region using adaptive background reconstruction.

    Optimized version:
    - Keeps the original full-resolution frame.
    - Analyses the surrounding background.
    - Flat/simple backgrounds use matched background colour.
    - Complex backgrounds use OpenCV inpainting.
    - Expensive inpainting and blending run only on a padded ROI.
    """

    x, y, w, h = bbox

    frame_h, frame_w = frame.shape[:2]

    # --------------------------------------------------
    # CLAMP SELECTED BOX
    # --------------------------------------------------

    x = max(
        0,
        min(int(x), frame_w - 1)
    )

    y = max(
        0,
        min(int(y), frame_h - 1)
    )

    w = max(
        1,
        min(int(w), frame_w - x)
    )

    h = max(
        1,
        min(int(h), frame_h - y)
    )

    # --------------------------------------------------
    # SAMPLE SURROUNDING BACKGROUND
    # --------------------------------------------------

    sample_size = 12

    samples = []

    # Above
    if y > 0:

        y0 = max(
            0,
            y - sample_size
        )

        region = frame[
            y0:y,
            x:min(frame_w, x + w)
        ]

        if region.size > 0:

            samples.append(
                region.reshape(-1, 3)
            )

    # Below
    if y + h < frame_h:

        y1 = min(
            frame_h,
            y + h + sample_size
        )

        region = frame[
            y + h:y1,
            x:min(frame_w, x + w)
        ]

        if region.size > 0:

            samples.append(
                region.reshape(-1, 3)
            )

    # Left
    if x > 0:

        x0 = max(
            0,
            x - sample_size
        )

        region = frame[
            y:min(frame_h, y + h),
            x0:x
        ]

        if region.size > 0:

            samples.append(
                region.reshape(-1, 3)
            )

    # Right
    if x + w < frame_w:

        x1 = min(
            frame_w,
            x + w + sample_size
        )

        region = frame[
            y:min(frame_h, y + h),
            x + w:x1
        ]

        if region.size > 0:

            samples.append(
                region.reshape(-1, 3)
            )

    # --------------------------------------------------
    # ANALYSE BACKGROUND COMPLEXITY
    # --------------------------------------------------

    if samples:

        surrounding_pixels = np.concatenate(
            samples,
            axis=0
        )

        background_color = np.median(
            surrounding_pixels,
            axis=0
        ).astype(np.uint8)

        complexity = float(
            np.mean(
                np.std(
                    surrounding_pixels.astype(
                        np.float32
                    ),
                    axis=0
                )
            )
        )

    else:

        background_color = np.array(
            [0, 0, 0],
            dtype=np.uint8
        )

        complexity = 999.0

    # --------------------------------------------------
    # CREATE PADDED ROI
    # --------------------------------------------------
    # Extra surrounding pixels give inpainting enough
    # neighbouring background information.

    roi_padding = max(
    16,
    int(feather)
)

    roi_x0 = max(
        0,
        x - roi_padding
    )

    roi_y0 = max(
        0,
        y - roi_padding
    )

    roi_x1 = min(
        frame_w,
        x + w + roi_padding
    )

    roi_y1 = min(
        frame_h,
        y + h + roi_padding
    )

    # Copy only the small working area.
    original_roi = frame[
        roi_y0:roi_y1,
        roi_x0:roi_x1
    ].copy()

    roi_mask = mask[
        roi_y0:roi_y1,
        roi_x0:roi_x1
    ].copy()

    # Coordinates of selected box inside ROI.
    local_x = x - roi_x0
    local_y = y - roi_y0

        # --------------------------------------------------
    # FLAT BACKGROUND
    # --------------------------------------------------

    if complexity < 28:

        repaired_roi = original_roi.copy()

        # --------------------------------------------------
        # LOCAL PLAIN-BACKGROUND RECONSTRUCTION
        # --------------------------------------------------
        # Estimate the background from the nearest clean
        # pixels around the selected watermark area.

        roi_h, roi_w = original_roi.shape[:2]

        x0 = local_x
        y0 = local_y
        x1 = min(roi_w, local_x + w)
        y1 = min(roi_h, local_y + h)

        patch_h = y1 - y0
        patch_w = x1 - x0

        top_color = None
        bottom_color = None
        left_color = None
        right_color = None

        edge_sample = 6

        # Top
        if y0 > 0:

            sy0 = max(
                0,
                y0 - edge_sample
            )

            top_strip = original_roi[
                sy0:y0,
                x0:x1
            ]

            if top_strip.size > 0:
                top_color = np.median(
                    top_strip.reshape(-1, 3),
                    axis=0
                )

        # Bottom
        if y1 < roi_h:

            sy1 = min(
                roi_h,
                y1 + edge_sample
            )

            bottom_strip = original_roi[
                y1:sy1,
                x0:x1
            ]

            if bottom_strip.size > 0:
                bottom_color = np.median(
                    bottom_strip.reshape(-1, 3),
                    axis=0
                )

        # Left
        if x0 > 0:

            sx0 = max(
                0,
                x0 - edge_sample
            )

            left_strip = original_roi[
                y0:y1,
                sx0:x0
            ]

            if left_strip.size > 0:
                left_color = np.median(
                    left_strip.reshape(-1, 3),
                    axis=0
                )

        # Right
        if x1 < roi_w:

            sx1 = min(
                roi_w,
                x1 + edge_sample
            )

            right_strip = original_roi[
                y0:y1,
                x1:sx1
            ]

            if right_strip.size > 0:
                right_color = np.median(
                    right_strip.reshape(-1, 3),
                    axis=0
                )

        # --------------------------------------------------
        # BUILD A SMOOTH COLOUR GRADIENT
        # --------------------------------------------------

        if (
            top_color is not None and
            bottom_color is not None
        ):

            vertical = np.linspace(
                top_color,
                bottom_color,
                patch_h,
                dtype=np.float32
            )

            vertical = np.repeat(
                vertical[:, None, :],
                patch_w,
                axis=1
            )

        else:

            vertical = np.full(
                (
                    patch_h,
                    patch_w,
                    3
                ),
                background_color,
                dtype=np.float32
            )

        if (
            left_color is not None and
            right_color is not None
        ):

            horizontal = np.linspace(
                left_color,
                right_color,
                patch_w,
                dtype=np.float32
            )

            horizontal = np.repeat(
                horizontal[None, :, :],
                patch_h,
                axis=0
            )

            reconstructed_patch = (
                vertical * 0.5 +
                horizontal * 0.5
            )

        else:

            reconstructed_patch = vertical

        reconstructed_patch = np.clip(
            reconstructed_patch,
            0,
            255
        ).astype(np.uint8)

        repaired_roi[
            y0:y1,
            x0:x1
        ] = reconstructed_patch

    # --------------------------------------------------
    # TRY RECONSTRUCTED BACKGROUND FIRST
    # --------------------------------------------------

    if background_roi is not None:

        if background_roi.shape == original_roi.shape:

            repaired_roi = background_roi.copy()

        else:

            repaired_roi = None
    else:

        repaired_roi = None

    # --------------------------------------------------
    # COMPLEX BACK+GROUND
    # --------------------------------------------------

    if repaired_roi is None:

        # Pass 1 - Telea
        telea = cv2.inpaint(
            original_roi,
            roi_mask,
            3,
            cv2.INPAINT_TELEA
        )
        
        # Pass 2 - Navier-Stokes
        ns = cv2.inpaint(
            original_roi,
            roi_mask,
            3,
            cv2.INPAINT_NS
        )

        # Blend both repairs
        repaired_roi = cv2.addWeighted(
            telea,
            0.6,
            ns,
            0.4,
            0
        )

    # --------------------------------------------------
    # FEATHER MASK EDGES INSIDE ROI ONLY
    # --------------------------------------------------

    feather_size = max(
        3,
        int(feather)
    )

    if feather_size % 2 == 0:
        feather_size += 1

    soft_mask = cv2.GaussianBlur(
        roi_mask,
        (
            feather_size,
            feather_size
        ),
        0
    )

    alpha = (
        soft_mask.astype(
            np.float32
        ) / 255.0
    )

    alpha = alpha[..., None]

    # --------------------------------------------------
    # EDGE-AWARE BLENDING
    # --------------------------------------------------

    gray = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 60, 120)

    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    edge_weight = 1.0 - (edges.astype(np.float32) / 255.0)

    edge_weight = edge_weight[..., None]

    alpha = alpha * edge_weight + alpha * 0.35

    alpha = np.clip(alpha, 0.0, 1.0)

    blended_roi = (
         repaired_roi.astype(np.float32) * alpha +
         original_roi.astype(np.float32) * (1.0 - alpha)
)
    
    blended_roi = np.clip(
        blended_roi,
        0,
        255
        ).astype(np.uint8)
    
    # --------------------------------------------------
    # LOCAL TEXTURE MATCHING
    # --------------------------------------------------

    # Estimate surrounding texture strength
    gray = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_32F)

    texture_strength = np.std(lap)

    # Very subtle texture generation
    noise = np.random.normal(
         0,
             texture_strength * 0.08,
             blended_roi.shape
             ).astype(np.float32)
    
    textured = blended_roi.astype(np.float32) + noise
    blended_roi = np.clip(
        textured,
        0,
    255
).astype(np.uint8)


    # --------------------------------------------------
    # COLOR HARMONIZATION
    # ---------------------------------------------------

    mask_bool = roi_mask > 0

    if np.any(mask_bool):

        repaired_pixels = blended_roi[mask_bool].astype(np.float32)

        surrounding_pixels = original_roi[~mask_bool].astype(np.float32)

        if len(surrounding_pixels) > 100:

            target_mean = surrounding_pixels.mean(axis=0)
            current_mean = repaired_pixels.mean(axis=0)

            correction = target_mean - current_mean

            repaired_pixels += correction * 0.35
            
            repaired_pixels = np.clip(
                repaired_pixels,
                0,
                255
            )

            blended_roi[mask_bool] = repaired_pixels.astype(np.uint8)

    # --------------------------------------------------
    # PUT REPAIRED ROI BACK INTO FULL FRAME
    # --------------------------------------------------

    output_frame = frame.copy()

    output_frame[
        roi_y0:roi_y1,
        roi_x0:roi_x1
    ] = blended_roi

    return output_frame

def process_video(
    input_path,
    logo_path=None,
    output_path=None,
    preview=False,
    padding=6,
    threshold=0.55,
       scales=(
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
    1.60,
    1.80,
    2.00,
),
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
    fully_auto_padding=22,
manual_box=None,
log=print,
progress_callback=None,

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
    import os

    print("=" * 60)
    print("INPUT PATH :", input_path)
    print("FILE EXISTS:", os.path.exists(input_path))
    print("FILE SIZE  :", os.path.getsize(input_path) if os.path.exists(input_path) else "N/A")
    print("=" * 60)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
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

        # ---------------------------------------------------------
    # Manual review mode
    # ---------------------------------------------------------
    if manual_box is not None:

        fixed_bbox = (
            max(0, int(manual_box["x"])),
            max(0, int(manual_box["y"])),
            int(manual_box["width"]),
            int(manual_box["height"])
        )

        detect_method = "manual_review"
        detect_confidence = 1.0

    elif fully_auto:
        print("ENTERED FULLY_AUTO")
        log("STEP 1: fully_auto entered")
        ret, first_frame = cap.read()
        log("STEP 2: first frame read")
        print("ENTERED FULLY_AUTO")

        if not ret:
            raise RuntimeError("Could not read first frame.")

        score, name, x, y, w, h = detect_best_region(first_frame)

        print("=" * 60)
        print("V2 DETECTED")
        print(f"Score : {score}")
        print(f"Name  : {name}")
        print(f"BBox  : {(x, y, w, h)}")
        print("=" * 60)
        print(f"[V2] score={score:.2f}")
        print(f"[V2] bbox={(x, y, w, h)}")

        log(f"STEP 3: detector finished: {(x, y, w, h)}")

        fixed_bbox = (x, y, w, h)

        detect_method = "logo_detector_v2"
        detect_confidence = min(1.0, score / 1000.0)

        log(f"  [V2] {name} score={score:.2f}")

        padding = fully_auto_padding

        cap.release()
        cap = cv2.VideoCapture(str(input_path))

    elif locked_position:
        if logo_path is None:
            raise ValueError(
                "locked_position=True requires logo_path (a crop of the logo)."
            )

        cap.release()

        fixed_bbox = detect_locked_position(
            input_path,
            logo_path,
            n_samples=locked_position_samples,
            log=log,
        )

        if fixed_bbox is None:
            raise RuntimeError(
                "Could not lock a reliable position for this logo. "
                "Try a tighter/cleaner logo crop, or fall back to "
                "per-frame matching (locked_position=False)."
            )

        log(f"  [locked-position] watermark region: {fixed_bbox} (x, y, w, h)")
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
    log(f"Writer opened: {writer.isOpened()}")

    inpaint_flag = cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS

    last_match = bbox_to_match_format(fixed_bbox) if fixed_bbox is not None else None

    # Use manually selected review box if provided
    if manual_box is not None:
        last_match = (
            (
                max(0, int(manual_box["x"])),
                max(0, int(manual_box["y"]))
            ),
            (
                max(1, int(manual_box["width"])),
                max(1, int(manual_box["height"]))
            )
        )

    frame_idx = 0
    matches_found = 0
    # Performance profiling
    total_mask_time = 0.0
    total_remove_time = 0.0
    total_write_time = 0.0

    log("DEBUG: Starting frame loop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Decide whether to run template matching on this frame
        if (
            manual_box is None
            and not auto_detect
            and not locked_position
            and not fully_auto
        ):
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
            # fixed region applies to every frame by definition
            matches_found += 1

        # Render preview or perform inpainting per frame
        if preview:
            out_frame = frame.copy()
            if last_match is not None:
                (x, y), (lw, lh) = last_match
                cv2.rectangle(out_frame, (x, y), (x + lw, y + lh), (0, 0, 255), 2)
        else:
            if last_match is not None:
                # --------------------------------------------------
                # MEASURE MASK CREATION
                # --------------------------------------------------
                mask_start = time.perf_counter()
                mask, _ = build_mask(
                    frame.shape,
                    last_match[0],
                    last_match[1],
                    padding
                )
                mask = refine_mask(frame, mask)
                total_mask_time += (
                    time.perf_counter() - mask_start
                )
                # --------------------------------------------------
                # MEASURE ADAPTIVE REMOVAL
                # --------------------------------------------------

                (x, y), (lw, lh) = last_match

                # Collect ROI for background learning
                # Use the current match bbox when no explicit bbox is available.
                w, h = lw, lh

                roi_padding = 20

                roi_x0 = max(0, x - roi_padding)
                roi_y0 = max(0, y - roi_padding)
                roi_x1 = min(frame.shape[1], x + w + roi_padding)
                roi_y1 = min(frame.shape[0], y + h + roi_padding)

                roi = frame[roi_y0:roi_y1, roi_x0:roi_x1]

                background_roi = None
                if background_builder is not None:
                    background_builder.add(roi)
                    # Build only after we have enough samples
                    if background_builder.count() >= 15:
                        background_roi = background_builder.build()

                remove_start = time.perf_counter()
                out_frame = adaptive_remove_region(
                    frame=frame,
                    mask=mask,
                    bbox=(x, y, lw, lh),
                    feather=15,
                    background_roi=background_roi,
                )

                total_remove_time += (
                    time.perf_counter() - remove_start
                )
            else:
                out_frame = frame

        write_start = time.perf_counter()
        writer.write(out_frame)
        frame_idx += 1
        # --------------------------------------------------
        # REPORT LIVE FRAME PROGRESS
        # --------------------------------------------------
        if progress_callback is not None:
            try:
                progress_callback(frame_idx, total_frames)
            except Exception as callback_error:
                log(
                    f"Progress callback error: "
                    f"{callback_error}"
                )
                if progress_every and frame_idx % progress_every == 0:
                    log(
                        f"    ...{frame_idx}/{total_frames} frames"
                    )

    log("DEBUG: Finished writing frames")
    log("=" * 60)

    cap.release()

    # Finalize the output video before muxing audio
    writer.release()
    import subprocess
    import os

    log(f"Output exists: {os.path.exists(tmp_video)}")

    if os.path.exists(tmp_video):
        size = os.path.getsize(tmp_video)
        log(f"Output size: {size} bytes")

        probe = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_format",
                "-show_streams",
                str(tmp_video),
            ],
            capture_output=True,
            text=True,
        )

        log(f"TMP ffprobe return code: {probe.returncode}")
        log(f"TMP ffprobe stdout:\n{probe.stdout}")
        log(f"TMP ffprobe stderr:\n{probe.stderr}")
    else:
        log(f"Output video missing: {tmp_video}")
        cv2.destroyAllWindows()
        raise RuntimeError(f"Output video missing: {tmp_video}")

    log("PERFORMANCE PROFILE")
    log(f"Frames processed : {frame_idx}")
    log(f"Mask creation    : {total_mask_time:.2f} seconds")
    log(f"Adaptive removal : {total_remove_time:.2f} seconds")
    log(f"Video writing    : {total_write_time:.2f} seconds")

    measured_total = (
        total_mask_time +
        total_remove_time +
        total_write_time
    )

    log(f"Measured total   : {measured_total:.2f} seconds")
    log("=" * 60)

    sampled = (
        max(1, frame_idx // sample_every + 1)
        if (not auto_detect and not locked_position and not fully_auto)
        else max(1, frame_idx)
)

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

    log("DEBUG: writer.release() Starting audio mux")
    import os
    log(f"Writer released")

    log(f"Output exists: {os.path.exists(tmp_video)}")

    if os.path.exists(tmp_video):
        size = os.path.getsize(tmp_video)
        log(f"Output size: {size} bytes")

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                str(tmp_video),
            ],
            capture_output=True,
            text=True,
        )

        log(f"TMP ffprobe return code: {probe.returncode}")
        log(f"TMP ffprobe stdout:\n{probe.stdout}")
        log(f"TMP ffprobe stderr:\n{probe.stderr}")

        if size == 0:
            raise RuntimeError("VideoWriter created a 0-byte output file.")
    else:
        raise RuntimeError(f"Output video not found: {tmp_video}")

    audio_status = mux_audio(
        input_path,
        tmp_video,
        output_path,
    )
    report["audio"] = audio_status
    log(f"DEBUG: Final MP4 validated successfully: {output_path}")
    if detect_rate < 0.3:
        report["status"] = "low_detection_warning"

    # ----------------------------------------------------
    # Create Before/After comparison video
    # ----------------------------------------------------
    try:
        comparison_output = str(output_path).replace(
            ".mp4",
            "_comparison.mp4",
        )
        create_side_by_side(
            str(input_path),
            str(output_path),
            comparison_output,
        )

        report["comparison"] = comparison_output
        log(f"Comparison video created: {comparison_output}")

    except Exception as e:
        log(f"Comparison video failed: {e}")

    if detect_rate < 0.3:
        report["status"] = "low_detection_warning"
            
    log("DEBUG: Returning report")

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
