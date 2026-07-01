import cv2
import numpy as np


def load_sample_frames(video_path, n_samples=30):
    """
    Load evenly spaced frames from a video.

    Returns:
        frames (list of BGR images)
        width
        height
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video contains no frames.")

    indices = np.linspace(
        0,
        total_frames - 1,
        min(n_samples, total_frames),
        dtype=int,
    )

    frames = []

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        ok, frame = cap.read()

        if ok:
            frames.append(frame)

    cap.release()

    return frames, width, height


def create_average_frame(frames):
    """
    Average all sampled frames.
    """

    stack = np.stack(frames).astype(np.float32)

    avg = np.mean(stack, axis=0)

    return avg.astype(np.uint8)


def edge_map(image):
    """
    Produce a clean edge image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        blur,
        50,
        150,
    )

    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=1)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return edges 
