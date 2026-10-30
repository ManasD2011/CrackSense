from camera import Camera
import cv2
import os

os.makedirs("output", exist_ok=True)

cam = Camera()

frame, ts = cam.read()

if frame is None:
    print("Camera failed")
else:
    print("Timestamp:", ts)
    print("Shape:", frame.shape)

    cv2.imwrite("output/frame.jpg", frame)
    print("Saved output/frame.jpg")

cam.release()
