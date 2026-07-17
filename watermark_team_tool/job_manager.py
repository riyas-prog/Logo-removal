import json
import shutil
import uuid
from pathlib import Path

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)


def create_job(video_path):

    job_id = uuid.uuid4().hex[:8]

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    src = Path(video_path)
    dst = job_dir / "original.mp4"

    shutil.copy2(src, dst)

    data = {
        "job_id": job_id,
        "filename": src.name,
        "status": "uploaded",

        "thumbnail": None,

        "width": None,
        "height": None,
        "fps": None,
        "duration": None,

        "logo_corner": None,
        "confidence": None,

        "suggested_box": None,
        "manual_box": None,

        "output": None
    }

    save_job(job_id, data)

    return job_id


def load_job(job_id):

    with open(JOBS_DIR / job_id / "analysis.json") as f:
        return json.load(f)


def save_job(job_id, data):

    with open(JOBS_DIR / job_id / "analysis.json", "w") as f:
        json.dump(data, f, indent=4)