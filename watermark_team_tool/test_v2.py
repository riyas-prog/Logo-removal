import cv2
from logo_detector_v2 import detect_best_region

cap = cv2.VideoCapture("test.mp4")

ret, frame = cap.read()

cap.release()

result = detect_best_region(frame)

print(result)

score, name, x, y, w, h = result

cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

cv2.imwrite("best_region.png", frame)

print("Saved best_region.png")