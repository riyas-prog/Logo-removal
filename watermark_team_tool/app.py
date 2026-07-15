#!/usr/bin/env python3
"""
app.py - Team Watermark Removal Tool

A small Flask web app wrapping the watermark-removal engine:
  - Bulk upload (multiple videos at once)
  - Fully automatic watermark detection (no logo crop needed) - falls back
    from fast static-region detection to bootstrapped template search for
    animated logos, same engine validated in core_processing.py
  - Simple password gate so this isn't wide open to the public
  - Per-video job status + download links once done

Run with:
    python3 app.py

Then open http://localhost:5000 (or your server's address) in a browser.

Set a real password via environment variable before running in any shared
environment:
    export TEAM_TOOL_PASSWORD="something-only-your-team-knows"
    python3 app.py

IMPORTANT: this app is intended for a small private team, not public
deployment. See README for hosting notes.
"""

import os
import json
import shutil
import cv2
import threading
import time
import traceback
import uuid
from pathlib import Path
import zipfile
import tempfile


from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from gallery_controller import get_gallery
from review_queue import get_review_jobs
from review_controller import get_review_session
from process_controller import process_job
from review_api import save_manual_box
from core_processing import process_video

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB total request size, adjust as needed

app = Flask(__name__)

@app.route("/ping")
def ping():
    return "PING WORKS"

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

TEAM_PASSWORD = os.environ.get("TEAM_TOOL_PASSWORD", "changeme")
print("DEBUG TEAM_PASSWORD =", repr(TEAM_PASSWORD))

# In-memory job store: {job_id: {filename, status, message, output_path, ...}}
# Fine for a small team tool on a single process; swap for a real DB/queue
# (e.g. Redis + RQ/Celery) if this needs to scale beyond a handful of
# concurrent users or survive app restarts mid-job.
JOBS = {}
JOBS_LOCK = threading.Lock()

BATCH_PROGRESS = {}
BATCH_PROGRESS_LOCK = threading.Lock()

def require_login():
    return session.get("authed") is True

def cleanup_old_files(max_age_hours=6):
    """
    Delete stale job folders, output videos, ZIP files,
    and temporary uploads older than max_age_hours.

    Recent and currently processing batches are protected.
    """

    cutoff_time = time.time() - (
        max_age_hours * 60 * 60
    )

    # --------------------------------------------------
    # PROTECT CURRENTLY ACTIVE JOBS / BATCHES
    # --------------------------------------------------

    active_job_ids = set()
    active_batch_ids = set()

    with BATCH_PROGRESS_LOCK:

        for batch_id, progress in BATCH_PROGRESS.items():

            if progress.get("status") in (
                "processing",
                "creating_zip"
            ):

                active_batch_ids.add(batch_id)

                for video in progress.get(
                    "videos",
                    []
                ):
                    job_id = video.get("job_id")

                    if job_id:
                        active_job_ids.add(job_id)

    # --------------------------------------------------
    # CLEAN OLD JOB FOLDERS
    # --------------------------------------------------

    jobs_dir = BASE_DIR / "jobs"

    if jobs_dir.exists():

        for job_dir in jobs_dir.iterdir():

            if not job_dir.is_dir():
                continue

            if job_dir.name in active_job_ids:
                continue

            try:

                if (
                    job_dir.stat().st_mtime <
                    cutoff_time
                ):
                    shutil.rmtree(
                        job_dir,
                        ignore_errors=True
                    )

                    print(
                        "CLEANED OLD JOB:",
                        job_dir
                    )

            except Exception as error:

                print(
                    "JOB CLEANUP ERROR:",
                    job_dir,
                    error
                )

    # --------------------------------------------------
    # CLEAN OLD OUTPUT FILES
    # --------------------------------------------------

    if OUTPUT_DIR.exists():

        for output_file in OUTPUT_DIR.iterdir():

            if not output_file.is_file():
                continue

            # Protect ZIP belonging to active batch
            is_active_batch_file = any(
                batch_id in output_file.name
                for batch_id in active_batch_ids
            )

            if is_active_batch_file:
                continue

            try:

                if (
                    output_file.stat().st_mtime <
                    cutoff_time
                ):
                    output_file.unlink(
                        missing_ok=True
                    )

                    print(
                        "CLEANED OLD OUTPUT:",
                        output_file
                    )

            except Exception as error:

                print(
                    "OUTPUT CLEANUP ERROR:",
                    output_file,
                    error
                )

    # --------------------------------------------------
    # CLEAN LEFTOVER TEMPORARY UPLOADS
    # --------------------------------------------------

    if UPLOAD_DIR.exists():

        for upload_file in UPLOAD_DIR.iterdir():

            if not upload_file.is_file():
                continue

            try:

                if (
                    upload_file.stat().st_mtime <
                    cutoff_time
                ):
                    upload_file.unlink(
                        missing_ok=True
                    )

                    print(
                        "CLEANED OLD UPLOAD:",
                        upload_file
                    )

            except Exception as error:

                print(
                    "UPLOAD CLEANUP ERROR:",
                    upload_file,
                    error
                )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == TEAM_PASSWORD:
            session["authed"] = True
            return redirect(url_for("index"))
        return render_template("login.html", error="Incorrect password")
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    if not require_login():
        return redirect(url_for("login"))
    return render_template("index.html")
    
@app.route("/gallery")
def gallery():

    gallery = get_gallery()

    return render_template(
        "gallery.html",
        gallery=gallery
    )

@app.route("/dashboard")
def dashboard():

    if not require_login():
        return redirect(url_for("login"))

    jobs = get_review_jobs()

    return render_template(
        "dashboard.html",
        jobs=jobs
    )

def _process_job(job_id, input_path, output_path):
    """Runs in a background thread; updates JOBS[job_id] as it progresses."""
    def log(msg):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["log"].append(str(msg))

    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "processing"

        report = process_video(
            input_path=input_path,
            output_path=output_path,
            fully_auto=True,
            log=log,
        )

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["report"] = {
                "detect_method": report.get("detect_method"),
                "detect_confidence": report.get("detect_confidence"),
                "detect_rate": report.get("detect_rate"),
                "frames": report.get("frames"),
                "audio": report.get("audio"),
            }
            if report.get("audio") == "ffmpeg_not_found":
                JOBS[job_id]["warning"] = (
                    "Watermark removed successfully, but the output has no audio - "
                    "ffmpeg isn't installed on this server. Install ffmpeg and "
                    "reprocess to keep audio."
                )
            elif report.get("audio") not in ("ok", "no_audio_in_source"):
                JOBS[job_id]["warning"] = (
                    "Watermark removed successfully, but audio could not be added "
                    "back to the output."
                )
            JOBS[job_id]["output_path"] = str(output_path)

        # Update analysis.json for review/download workflow
        analysis_file = BASE_DIR / "jobs" / job_id / "analysis.json"

        if analysis_file.exists():
            with open(analysis_file, "r") as f:
                analysis = json.load(f)

            analysis["status"] = "done"
            analysis["output"] = str(output_path)

            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=4)

        # Delete uploaded file after successful processing
        Path(input_path).unlink(missing_ok=True)

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
        log(f"[error] {e}")
        log(traceback.format_exc())

        # Delete uploaded file if processing failed
        Path(input_path).unlink(missing_ok=True)


@app.route("/upload", methods=["POST"])
def upload():
    if not require_login():
        return jsonify({"error": "Not authenticated"}), 401

    files = request.files.getlist("videos")

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    batch_id = uuid.uuid4().hex[:12]
    job_ids = []

    for f in files:
        if not f.filename:
            continue

        ext = Path(f.filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            continue

        job_id = uuid.uuid4().hex[:12]
        safe_name = f"{job_id}{ext}"

        input_path = UPLOAD_DIR / safe_name

        f.save(str(input_path))

        # Create review job folder
        job_dir = BASE_DIR / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Copy uploaded video into job folder
        shutil.copy2(
            input_path,
            job_dir / "original.mp4"
        )

        # Temporary upload copy is no longer needed.
        # The working copy now exists inside jobs/<job_id>/original.mp4.
        input_path.unlink(
        missing_ok=True
)
        # ---------------------------------------------------
        # Create thumbnail
        # ---------------------------------------------------
        thumbnail_path = job_dir / "original.jpg"
        cap = cv2.VideoCapture(str(job_dir / "original.mp4"))
        ret = False
        frame = None
        # Try several frames instead of only the first frame
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                break

        cap.release()
        if ret and frame is not None:
            cv2.imwrite(str(thumbnail_path), frame)
            print(
                "THUMBNAIL:",
                job_id,
                "PATH:",
                thumbnail_path
            )
        else:
            print(
                "WARNING: Could not create thumbnail for:",
                f.filename
            )

         # Save review information
        analysis = {
            "job_id": job_id,
            "batch_id": batch_id,
            "filename": f.filename,
            "status": "queued",
            "thumbnail": "original.jpg",
            "manual_box": None,
            "output": None,
            "created_at": time.time()
        }

        with open(
            job_dir / "analysis.json",
            "w"
        ) as fp:
            json.dump(
                analysis,
                fp,
                indent=4
            )

        # In-memory job
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "batch_id": batch_id,
                "original_filename": f.filename,
                "status": "queued",
                "log": [],
                "error": None,
                "warning": None,
                "report": None,
                "output_path": None,
                "created_at": time.time(),
            }

        job_ids.append(job_id)

    if not job_ids:
        return jsonify({
            "error": "No valid video files uploaded"
        }), 400

    return jsonify({
        "ok": True,
        "job_ids": job_ids,
        "batch_id": batch_id,
        "redirect": f"/review-session?batch_id={batch_id}&index=0"
    })


@app.route("/status/<job_id>")
def status(job_id):
    if not require_login():
        return jsonify({"error": "Not authenticated"}), 401

    with JOBS_LOCK:
        job = JOBS.get(job_id)

        if not job:
            return jsonify({"error": "Unknown job"}), 404

        return jsonify({
            "id": job["id"],
            "original_filename": job["original_filename"],
            "status": job["status"],
            "log": job["log"][-20:],
            "error": job["error"],
            "warning": job.get("warning"),
            "report": job["report"],
        })


@app.route("/thumbnail/<job_id>")
def thumbnail(job_id):

    print("***** THUMBNAIL ROUTE CALLED *****")
    print("JOB ID:", job_id)

    image_path = BASE_DIR / "jobs" / job_id / "original.jpg"
    print("IMAGE PATH:", image_path)

    if not image_path.exists():
        print("FILE NOT FOUND")
        return "Thumbnail not found", 404

    print("SENDING IMAGE")
    return send_file(image_path)

@app.route("/download/<job_id>")
def download(job_id):

    if not require_login():
        return redirect(url_for("login"))

    analysis_file = BASE_DIR / "jobs" / job_id / "analysis.json"

    if not analysis_file.exists():
        return "Job not found", 404

    with open(analysis_file, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    if analysis.get("status") != "done":
        return "Video is not ready yet", 404

    output_name = analysis.get("output")

    if not output_name:
        return "Output path missing from analysis", 404

    # process_controller stores only:
    # aa770879f744_clean.mp4
    output_path = OUTPUT_DIR / Path(output_name).name

    print("=" * 60)
    print("DOWNLOAD JOB:", job_id)
    print("ANALYSIS OUTPUT:", output_name)
    print("LOOKING FOR:", output_path)
    print("EXISTS:", output_path.exists())
    print("=" * 60)

    if not output_path.exists():
        return f"Output file missing: {output_path}", 404

    original_stem = Path(
        analysis.get("filename", job_id)
    ).stem

    return send_file(
        str(output_path),
        as_attachment=True,
        download_name=f"{original_stem}_clean.mp4"
    )


@app.route("/cleanup/<job_id>", methods=["POST"])
def cleanup(job_id):
    """Optional: delete a job's files once downloaded, to save disk space."""
    if not require_login():
        return jsonify({"error": "Not authenticated"}), 401
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)
    if job and job.get("output_path"):
        Path(job["output_path"]).unlink(missing_ok=True)
    return jsonify({"ok": True})

@app.route("/review/<job_id>", methods=["GET", "POST"])
def review(job_id):

    analysis_file = BASE_DIR / "jobs" / job_id / "analysis.json"

    if not analysis_file.exists():
        return "Job not found", 404

    with open(analysis_file, "r") as f:
        analysis = json.load(f)

    if request.method == "POST":
        return redirect(url_for("download", job_id=job_id))

    return render_template(
        "review.html",
        job=analysis,
        image_url=url_for("thumbnail", job_id=job_id),
    )
@app.route("/save-box", methods=["POST"])
def save_box():

    if not require_login():
        return jsonify({"ok": False}), 401

    data = request.get_json()

    save_manual_box(
        data["job_id"],
        {
            "x": data["x"],
            "y": data["y"],
            "width": data["width"],
            "height": data["height"]
        }
    )

    return jsonify({"ok": True})

@app.route("/review-session")
def review_session():

    if not require_login():
        return redirect(url_for("login"))

    # ---------------------------------------------------
    # GET CURRENT BATCH
    # ---------------------------------------------------
    batch_id = request.args.get("batch_id")

    if not batch_id:
        return "Missing batch_id", 400

    # ---------------------------------------------------
    # GET INDEX
    # ---------------------------------------------------
    try:
        index = int(request.args.get("index", 0))

    except (TypeError, ValueError):
        index = 0

    # ---------------------------------------------------
    # LOAD ONLY JOBS FROM THIS BATCH
    # ---------------------------------------------------
    jobs = []

    jobs_dir = BASE_DIR / "jobs"

    if jobs_dir.exists():

        for folder in jobs_dir.iterdir():

            if not folder.is_dir():
                continue

            analysis_file = folder / "analysis.json"

            if not analysis_file.exists():
                continue

            try:

                with open(
                    analysis_file,
                    "r"
                ) as f:

                    job = json.load(f)

                # ONLY CURRENT UPLOAD BATCH
                if job.get("batch_id") == batch_id:
                    jobs.append(job)

            except Exception as e:

                print(
                    "Could not load review job:",
                    analysis_file,
                    e
                )

    # ---------------------------------------------------
    # SORT IN ORIGINAL UPLOAD ORDER
    # ---------------------------------------------------
    jobs.sort(
        key=lambda job: job.get(
            "created_at",
            0
        )
    )

    if not jobs:
        return "No videos found for this review session", 404

    total = len(jobs)

    # Keep index valid
    index = max(
        0,
        min(
            index,
            total - 1
        )
    )

    session_data = {
        "current": index,
        "total": total,
        "batch_id": batch_id,
        "jobs": jobs
    }

    return render_template(
        "review_session.html",
        session=session_data
    )
@app.route("/process-job", methods=["POST"])
def process_job_route():

    if not require_login():
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    job_id = data["job_id"]

    print("PROCESS JOB:", job_id)

    output = process_job(job_id)

    print("OUTPUT:", output)

    return jsonify({
        "ok": True,
        "output": str(output)
    })

def _run_batch_process(batch_id):

    print("=" * 60)
    print("BACKGROUND BATCH STARTED:", batch_id)
    print("=" * 60)

    batch_jobs = []

    # --------------------------------------------------
    # FIND ALL JOBS IN THIS BATCH
    # --------------------------------------------------

    with JOBS_LOCK:

        for job_id, job_data in JOBS.items():

            if job_data.get("batch_id") == batch_id:

                batch_jobs.append({
                    "job_id": job_id,
                    "upload_index": job_data.get(
                        "upload_index",
                        job_data.get("index", 0)
                    )
                })

    batch_jobs.sort(
        key=lambda item: item["upload_index"]
    )

    total = len(batch_jobs)

    # --------------------------------------------------
    # INITIAL PROGRESS STATE
    # --------------------------------------------------

    with BATCH_PROGRESS_LOCK:

        BATCH_PROGRESS[batch_id] = {
            "status": "processing",
            "total": total,
            "completed": 0,
            "current": 0,
            "current_job_id": None,
            "videos": [
                {
                    "job_id": item["job_id"],
                    "number": index + 1,
                    "status": "waiting"
                }
                for index, item in enumerate(batch_jobs)
            ],
            "zip_ready": False,
            "error": None
        }

    results = []

    # --------------------------------------------------
    # PROCESS VIDEOS ONE BY ONE
    # --------------------------------------------------

    for index, item in enumerate(batch_jobs):

        job_id = item["job_id"]

        with BATCH_PROGRESS_LOCK:

            progress = BATCH_PROGRESS[batch_id]

            progress["current"] = index + 1
            progress["current_job_id"] = job_id

            progress["videos"][index]["status"] = (
                "processing"
            )

        print("-" * 60)
        print(
            f"PROCESSING {index + 1}/{total}:",
            job_id
        )
        print("-" * 60)

        try:
            # --------------------------------------------------
            # LIVE FRAME PROGRESS CALLBACK
            # --------------------------------------------------
            def update_frame_progress(
                current_frame,
                total_frames,
                batch_id=batch_id,
                video_index=index
            ):
                with BATCH_PROGRESS_LOCK:
                    progress = BATCH_PROGRESS.get(batch_id)

                    if progress is None:
                        return

                    video = progress["videos"][video_index]
                    video["current_frame"] = current_frame
                    video["total_frames"] = total_frames

                    if total_frames > 0:
                        video["percent"] = round(
                            (
                                current_frame /
                                total_frames
                            ) * 100
                        )
                    else:
                        video["percent"] = 0

            output = process_job(
                job_id,
                progress_callback=update_frame_progress
            )

            results.append({
                "job_id": job_id,
                "ok": True,
                "output": str(output)
            })

            with BATCH_PROGRESS_LOCK:
                progress = BATCH_PROGRESS[batch_id]
                progress["videos"][index]["status"] = "completed"
                progress["completed"] = index + 1
        except Exception as exc:
            print(f"ERROR PROCESSING {job_id}: {exc}")

            results.append({
                "job_id": job_id,
                "ok": False,
                "error": str(exc)
            })

            with BATCH_PROGRESS_LOCK:
                progress = BATCH_PROGRESS[batch_id]
                progress["videos"][index]["status"] = "failed"
                progress["completed"] = index + 1

            continue

    # --------------------------------------------------
    # CREATE ZIP
    # --------------------------------------------------

    successful_results = [
        result
        for result in results
        if result["ok"]
    ]

    if not successful_results:

        with BATCH_PROGRESS_LOCK:

            BATCH_PROGRESS[batch_id]["status"] = (
                "failed"
            )

            BATCH_PROGRESS[batch_id]["error"] = (
                "All videos failed to process."
            )

        return

    with BATCH_PROGRESS_LOCK:

        BATCH_PROGRESS[batch_id]["status"] = (
            "creating_zip"
        )

    zip_path = (
        BASE_DIR /
        "outputs" /
        f"batch_{batch_id}_cleaned.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        for result in successful_results:

            output_path = Path(
                result["output"]
            )

            if output_path.exists():

                zip_file.write(
                    output_path,
                    arcname=output_path.name
                )

    print(
        "BACKGROUND ZIP CREATED:",
        zip_path
    )

    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    with BATCH_PROGRESS_LOCK:

        progress = BATCH_PROGRESS[batch_id]

        progress["status"] = "completed"
        progress["zip_ready"] = True
        progress["download_url"] = (
            f"/batch-download/{batch_id}"
        )

    print("=" * 60)
    print("BACKGROUND BATCH COMPLETE")
    print("=" * 60)
    
@app.route("/batch-process")
def batch_process_route():

    if not require_login():
        return redirect("/login")

    batch_id = request.args.get("batch_id")

    if not batch_id:
        return jsonify({
            "ok": False,
            "error": "Missing batch_id"
        }), 400

    # Prevent accidental duplicate processing
    with BATCH_PROGRESS_LOCK:

        existing = BATCH_PROGRESS.get(batch_id)

        if existing and existing.get("status") in (
            "processing",
            "creating_zip"
        ):

            return jsonify({
                "ok": True,
                "batch_id": batch_id,
                "already_running": True
            })

    worker = threading.Thread(
        target=_run_batch_process,
        args=(batch_id,),
        daemon=True
    )

    worker.start()

    return jsonify({
        "ok": True,
        "batch_id": batch_id,
        "started": True
    })


@app.route("/batch-status/<batch_id>")
def batch_status_route(batch_id):

    if not require_login():
        return jsonify({
            "ok": False,
            "error": "Not authenticated"
        }), 401

    with BATCH_PROGRESS_LOCK:

        progress = BATCH_PROGRESS.get(
            batch_id
        )

        if progress is None:

            return jsonify({
                "ok": False,
                "error": "Batch progress not found"
            }), 404

        # Return a safe copy
        return jsonify({
            "ok": True,
            **progress
        })


# --------------------------------------------------
# DELAYED BATCH CLEANUP
# --------------------------------------------------

def cleanup_batch_after_delay(
    batch_id,
    batch_job_ids,
    zip_path,
    delay_seconds=3600
):

    print(
        f"CLEANUP SCHEDULED FOR BATCH {batch_id} "
        f"IN {delay_seconds} SECONDS"
    )

    # Wait 1 hour
    time.sleep(delay_seconds)

    print("=" * 60)
    print("STARTING DELAYED CLEANUP:", batch_id)
    print("=" * 60)

    # --------------------------------------------------
    # DELETE JOB FOLDERS
    # jobs/<job_id>/
    # --------------------------------------------------

    for job_id in batch_job_ids:

        job_dir = (
            BASE_DIR /
            "jobs" /
            job_id
        )

        try:

            if job_dir.exists():

                shutil.rmtree(
                    job_dir,
                    ignore_errors=True
                )

                print(
                    "DELETED JOB FOLDER:",
                    job_dir
                )

        except Exception as exc:

            print(
                "JOB FOLDER CLEANUP ERROR:",
                job_id,
                exc
            )

        # --------------------------------------------------
        # DELETE INDIVIDUAL CLEANED OUTPUTS
        # --------------------------------------------------

        try:

            output_dir = (
                BASE_DIR /
                "outputs"
            )

            for output_file in output_dir.glob(
                f"{job_id}*"
            ):

                if output_file.is_file():

                    output_file.unlink(
                        missing_ok=True
                    )

                    print(
                        "DELETED OUTPUT:",
                        output_file
                    )

        except Exception as exc:

            print(
                "OUTPUT CLEANUP ERROR:",
                job_id,
                exc
            )

    # --------------------------------------------------
    # DELETE BATCH ZIP
    # --------------------------------------------------

    try:

        zip_path.unlink(
            missing_ok=True
        )

        print(
            "DELETED BATCH ZIP:",
            zip_path
        )

    except Exception as exc:

        print(
            "ZIP CLEANUP ERROR:",
            exc
        )

    # --------------------------------------------------
    # REMOVE JOBS FROM MEMORY
    # --------------------------------------------------

    with JOBS_LOCK:

        for job_id in batch_job_ids:

            JOBS.pop(
                job_id,
                None
            )

    # --------------------------------------------------
    # REMOVE BATCH PROGRESS FROM MEMORY
    # --------------------------------------------------

    with BATCH_PROGRESS_LOCK:

        BATCH_PROGRESS.pop(
            batch_id,
            None
        )

    print("=" * 60)
    print(
        "DELAYED BATCH CLEANUP COMPLETE:",
        batch_id
    )
    print("=" * 60)


@app.route("/batch-download/<batch_id>")
def batch_download_route(batch_id):

    if not require_login():
        return redirect("/login")

    zip_path = (
        BASE_DIR /
        "outputs" /
        f"batch_{batch_id}_cleaned.zip"
    )

    if not zip_path.exists():

        return "Batch ZIP not found", 404

    # --------------------------------------------------
    # FIND JOBS BELONGING ONLY TO THIS BATCH
    # --------------------------------------------------

    batch_job_ids = []

    with JOBS_LOCK:

        for job_id, job_data in JOBS.items():

            if job_data.get("batch_id") == batch_id:

                batch_job_ids.append(
                    job_id
                )

    print(
        "DOWNLOAD STARTED FOR BATCH:",
        batch_id
    )

    print(
        "BATCH JOBS:",
        batch_job_ids
    )

    # --------------------------------------------------
    # START 1-HOUR CLEANUP TIMER
    # --------------------------------------------------

    cleanup_thread = threading.Thread(
        target=cleanup_batch_after_delay,
        args=(
            batch_id,
            batch_job_ids,
            zip_path,
            3600
        ),
        daemon=True
    )

    cleanup_thread.start()

    print(
        "1-HOUR CLEANUP TIMER STARTED:",
        batch_id
    )

    # --------------------------------------------------
    # DOWNLOAD NORMALLY
    # NO FILE IS DELETED DURING DOWNLOAD
    # --------------------------------------------------

    return send_file(
        zip_path,
        as_attachment=True,
        download_name=(
            f"batch_{batch_id}_cleaned.zip"
        )
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
