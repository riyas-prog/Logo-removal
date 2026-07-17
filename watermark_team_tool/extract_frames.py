import cv2
import os

VIDEO = "test.mp4"

cap = cv2.VideoCapture(VIDEO)

os.makedirs("frames", exist_ok=True)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(10):
    frame_no = int(i * max(1, total - 1) / 9)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"frames/frame_{i}.png", frame)

cap.release()

print("Frames extracted.")