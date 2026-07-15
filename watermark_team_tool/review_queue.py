import json
from pathlib import Path

JOBS_DIR = Path("jobs")


def get_review_jobs():

    jobs = []

    if not JOBS_DIR.exists():
        return jobs

    for folder in sorted(JOBS_DIR.iterdir()):

        if not folder.is_dir():
            continue

        info = folder / "analysis.json"

        if not info.exists():
            continue

        with open(info) as f:
            job = json.load(f)

        if job.get("status") == "queued":
            jobs.append(job)

    jobs.sort(
    key=lambda x: Path(JOBS_DIR / x["job_id"]).stat().st_mtime,
    reverse=True
)

    return jobs