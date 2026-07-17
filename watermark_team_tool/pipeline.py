from pathlib import Path

from job_manager import (
    create_job,
    load_job,
    save_job
)

from thumbnail_generator import create_thumbnail
from analysis_engine import analyze_video


def build_job(video_path):

    job_id = create_job(video_path)

    job = load_job(job_id)

    job_folder = Path("jobs") / job_id

    thumb = create_thumbnail(
        job_folder / "original.mp4",
        job_folder
    )

    info = analyze_video(
        job_folder / "original.mp4"
    )

    job["thumbnail"] = thumb

    job["width"] = info.width
    job["height"] = info.height
    job["fps"] = info.fps
    job["duration"] = info.duration

    job["status"] = "ready_for_review"

    save_job(job_id, job)

    return job_id