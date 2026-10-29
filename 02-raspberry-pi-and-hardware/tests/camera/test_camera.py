from camera import Camera
import cv2

cam = Camera()

while True:

    frame, ts = cam.read()

    if frame is None:
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
