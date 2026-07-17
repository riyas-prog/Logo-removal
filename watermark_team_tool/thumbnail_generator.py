import cv2
from pathlib import Path


def create_thumbnail(video_path, output_folder="static/thumbnails"):

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    ok, frame = cap.read()

    cap.release()

    if not ok:
        return None

    name = Path(video_path).stem + ".jpg"

    output = output_folder / name

    cv2.imwrite(str(output), frame)

    return output.name