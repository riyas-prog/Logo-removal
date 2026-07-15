import json
from pathlib import Path


def save_manual_box(job_id, box):

    file = Path("jobs") / job_id / "analysis.json"

    with open(file, "r") as f:
        data = json.load(f)

    data["manual_box"] = box

    with open(file, "w") as f:
        json.dump(data, f, indent=4)

    return True