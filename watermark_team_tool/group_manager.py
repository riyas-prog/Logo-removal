from dataclasses import dataclass
from collections import defaultdict


@dataclass
class VideoFingerprint:
    filename: str
    width: int
    height: int
    fps: float
    corner: str
    logo_width: int
    logo_height: int


def create_fingerprint(
    filename,
    width,
    height,
    fps,
    corner,
    logo_width,
    logo_height,
):
    return VideoFingerprint(
        filename=filename,
        width=width,
        height=height,
        fps=round(fps),
        corner=corner,
        logo_width=logo_width,
        logo_height=logo_height,
    )


def group_videos(fingerprints):
    """
    Group videos by similar characteristics.
    """

    groups = defaultdict(list)

    for fp in fingerprints:

        size_bucket = (
            round(fp.logo_width / 20),
            round(fp.logo_height / 20),
        )

        key = (
            fp.width,
            fp.height,
            fp.corner,
            size_bucket,
        )

        groups[key].append(fp)

    return groups