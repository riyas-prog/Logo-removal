import cv2
import numpy as np


class BackgroundBuilder:

    def __init__(self, max_frames=30):

        self.max_frames = max_frames
        self.samples = []

    def is_similar(self, roi, threshold=18.0):
        if not self.samples:
            return True
        
        previous = self.samples[-1]["gray"]
        current = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
    )
        
        if previous.shape != current.shape:
            return False
        
        difference = cv2.absdiff(
            previous,
            current
    )
        
        score = float(np.mean(difference))
        return score < threshold


    def add(self, roi):

        if roi is None or roi.size == 0:
            return
        if not self.is_similar(roi):
            return

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        sample = {
            "roi": roi.astype(np.float32),
            "gray": gray,
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
        }

        self.samples.append(sample)

        if len(self.samples) > self.max_frames:
            self.samples.pop(0)

    def build(self):

        if len(self.samples) < 5:
            return None

        stack = np.stack(
            [s["roi"] for s in self.samples],
            axis=0,
        )

        background = np.median(
            stack,
            axis=0,
        )

        return background.astype(np.uint8)

    def clear(self):
        self.samples.clear()

    def count(self):
        return len(self.samples)