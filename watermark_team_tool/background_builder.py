import cv2
import numpy as np


class BackgroundBuilder:

    def __init__(self):
        self.frames = []

    def add(self, frame):
        self.frames.append(frame.astype(np.float32))

    def build(self):

        if not self.frames:
            return None

        stack = np.stack(self.frames)

        background = np.median(
            stack,
            axis=0
        )

        return background.astype(np.uint8)