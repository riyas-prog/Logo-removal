import cv2
import numpy as np


def refine_mask(frame, mask):
    """
    Refine a coarse rectangular mask using image edges.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 40, 120)

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    refined = cv2.bitwise_and(
        mask,
        edges
    )

    refined = cv2.dilate(
        refined,
        np.ones((5, 5), np.uint8),
        iterations=1
    )

    refined = cv2.GaussianBlur(
        refined,
        (11, 11),
        0
    )

    return refined