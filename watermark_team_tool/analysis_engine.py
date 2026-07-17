from dataclasses import dataclass
from pathlib import Path
import cv2


@dataclass
class VideoAnalysis:
    filename: str
    filepath: str

    width: int
    height: int

    fps: float
    frame_count: int
    duration: float


def analyze_video(video_path):

    video_path = str(video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = 0

    if fps > 0:
        duration = frame_count / fps

    cap.release()

    return VideoAnalysis(
        filename=Path(video_path).name,
        filepath=video_path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
    )