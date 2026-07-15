import json
from pathlib import Path

from core_processing import process_video


BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_job(job_id, progress_callback=None):

    job_dir = JOBS_DIR / job_id
    analysis_file = job_dir / "analysis.json"

    if not analysis_file.exists():
        raise FileNotFoundError(
            f"Analysis file not found: {analysis_file}"
        )

    with open(analysis_file, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    manual_box = analysis.get("manual_box")

    if not manual_box:
        raise ValueError(
            f"No manual box saved for job {job_id}"
        )

    output_path = OUTPUT_DIR / f"{job_id}_clean.mp4"

    print("=" * 60)
    print("PROCESSING JOB:", job_id)
    print("INPUT:", job_dir / "original.mp4")
    print("OUTPUT:", output_path)
    print("MANUAL BOX:", manual_box)
    print("=" * 60)

    process_video(
    input_path=str(job_dir / "original.mp4"),
    output_path=str(output_path),
    manual_box=manual_box,
    auto_detect=False,
    progress_callback=progress_callback
)

    # IMPORTANT: verify processing actually created the file
    if not output_path.exists():
        raise RuntimeError(
            f"Processing finished but output file was not created: "
            f"{output_path}"
        )

    print("OUTPUT CREATED:", output_path)
    print("OUTPUT SIZE:", output_path.stat().st_size)

    analysis["status"] = "done"

    # Store only the filename, not another relative/absolute path
    analysis["output"] = output_path.name

    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=4)

    return str(output_path)