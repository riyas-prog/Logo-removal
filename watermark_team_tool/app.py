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
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify

from core_processing import process_video

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB total request size, adjust as needed

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

TEAM_PASSWORD = os.environ.get("TEAM_TOOL_PASSWORD", "changeme")
print("Loaded TEAM_PASSWORD =", repr(TEAM_PASSWORD))

# In-memory job store: {job_id: {filename, status, message, output_path, ...}}
# Fine for a small team tool on a single process; swap for a real DB/queue
# (e.g. Redis + RQ/Celery) if this needs to scale beyond a handful of
# concurrent users or survive app restarts mid-job.
JOBS = {}
JOBS_LOCK = threading.Lock()


def require_login():
    return session.get("authed") is True


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
        output_path = OUTPUT_DIR / f"{job_id}_clean{ext}"

        f.save(str(input_path))

        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "original_filename": f.filename,
                "status": "queued",
                "log": [],
                "error": None,
                "warning": None,
                "report": None,
                "output_path": None,
                "created_at": time.time(),
            }

        thread = threading.Thread(
            target=_process_job, args=(job_id, input_path, output_path), daemon=True
        )
        thread.start()
        job_ids.append(job_id)

    return jsonify({"job_ids": job_ids})


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
            "log": job["log"][-20:],  # last 20 lines is plenty for a progress view
            "error": job["error"],
            "warning": job.get("warning"),
            "report": job["report"],
        })


@app.route("/download/<job_id>")
def download(job_id):
    if not require_login():
        return redirect(url_for("login"))
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job["status"] != "done" or not job["output_path"]:
        return "Not ready", 404

    original_stem = Path(job["original_filename"]).stem
    ext = Path(job["output_path"]).suffix
    download_name = f"{original_stem}_clean{ext}"
    return send_file(job["output_path"], as_attachment=True, download_name=download_name)


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
