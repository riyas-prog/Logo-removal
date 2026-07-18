import cv2
import numpy as np
from pathlib import Path

class BackgroundBuilder:

    def __init__(self, max_frames=30):

        self.max_frames = max_frames
        self.samples = []
        self.debug_dir = Path("debug_backgrounds")
        self.debug_dir.mkdir(exist_ok=True)

        self.debug_index = 0

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
    
    def align_roi(self, reference, current):
        """

        Align the current ROI to the reference ROI using ECC.
        Returns the aligned ROI. If alignment fails,
        returns the original ROI.
        """
        try:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

            warp_matrix = np.eye(2, 3, dtype=np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)

            cv2.findTransformECC(ref_gray, cur_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria)

            aligned = cv2.warpAffine(
                current,
                warp_matrix,
                (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )

            return aligned
        except cv2.error:
            return current
            
    def add(self, roi):

        if roi is None or roi.size == 0:
            return
        if not self.is_similar(roi):
            return
        
        if self.samples:
            reference = self.samples[-1]["roi"].astype(np.uint8)

            roi = self.align_roi(
        reference,
        roi
    )


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

        background = background.astype(np.uint8)

        # Save every 30th reconstruction
        if self.debug_index % 30 == 0:

            filename = self.debug_dir / f"background_{self.debug_index:04d}.png"

            cv2.imwrite(
                str(filename),
                background
            )

        self.debug_index += 1
        print(f"BackgroundBuilder: built with {len(self.samples)} samples")

        return background

    def clear(self):
        self.samples.clear()

    def count(self):
        return len(self.samples)