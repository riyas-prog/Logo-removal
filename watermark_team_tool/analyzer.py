from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoAnalysis:
    filename: str
    filepath: str

    width: int
    height: int

    fps: float
    frame_count: int
    duration: float

    detected_corner: str

    logo_x: int
    logo_y: int
    logo_w: int
    logo_h: int

    detection_score: float