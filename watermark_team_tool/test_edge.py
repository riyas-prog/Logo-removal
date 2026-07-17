import cv2
import numpy as np

VIDEO = "test.mp4"   # <-- replace with your video

cap = cv2.VideoCapture(VIDEO)

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (360, 640))
    frames.append(frame.astype(np.float32))

cap.release()

avg = np.mean(frames, axis=0).astype(np.uint8)

gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 80, 180)

cv2.imwrite("average.png", avg)
cv2.imwrite("edges.png", edges)

print("Done.")