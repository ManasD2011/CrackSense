from camera import Camera
from optical_flow import OpticalFlow

cam = Camera()

flow = OpticalFlow()

print("\n===== Optical Flow Test =====\n")

while True:

    frame, timestamp = cam.read()

    result = flow.process(frame)

    if flow.initialized:
        break

cam.release()

print("\nInitialization Successful")
