from pathlib import Path
import json

from core_processing import process_video


def process_reviewed_job(job_id):
    job_dir = Path("jobs") / job_id

    analysis = job_dir / "analysis.json"

    with open(analysis, "r") as f:
        data = json.load(f)

    box = data["manual_box"]

    input_video = job_dir / "original.mp4"

    output_video = Path("outputs") / f"{job_id}_clean.mp4"

    process_video(
        str(input_video),
        str(output_video),
        box
    )

    data["status"] = "done"
    data["output"] = str(output_video)

    with open(analysis, "w") as f:
        json.dump(data, f, indent=4)

    return output_video