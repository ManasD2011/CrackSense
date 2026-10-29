"""
camera.py

Camera interface for the VIO project.

Features:
- Works on Raspberry Pi (V4L2)
- Works on Windows (DirectShow)
- Timestamp for every frame
- Optional camera calibration
- Self-test mode
"""

import cv2
import time
import platform
import os


class Camera:

    def __init__(
        self,
        camera_index=2,
        width=640,
        height=480,
        fps=30,
        camera_matrix=None,
        dist_coeffs=None
    ):

        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.K = camera_matrix
        self.dist = dist_coeffs

        system = platform.system()

        if system == "Windows":

            self.cap = cv2.VideoCapture(
                camera_index,
                cv2.CAP_DSHOW
            )

        else:

            device = f"/dev/video{camera_index}"

            self.cap = cv2.VideoCapture(
                device,
                cv2.CAP_V4L2
            )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open camera {camera_index}"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            fps
        )

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None, None

        timestamp = time.perf_counter()

        if self.K is not None and self.dist is not None:

            frame = cv2.undistort(
                frame,
                self.K,
                self.dist
            )

        return frame, timestamp

    def info(self):

        return {

            "Width":
                int(
                    self.cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                ),

            "Height":
                int(
                    self.cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                ),

            "FPS":
                self.cap.get(
                    cv2.CAP_PROP_FPS
                ),

        }

    def release(self):

        if self.cap is not None:

            self.cap.release()


# ---------------------------------------------------
# Self Test
# ---------------------------------------------------

if __name__ == "__main__":

    print("\n========== CAMERA TEST ==========\n")

    try:

        cam = Camera()

        info = cam.info()

        print("Camera opened successfully")

        print(f"Resolution : {info['Width']} x {info['Height']}")
        print(f"FPS        : {info['FPS']:.1f}")

        frame, ts = cam.read()

        if frame is None:

            print("ERROR : Unable to capture frame")

        else:

            print("Frame Capture : PASS")
            print(f"Timestamp     : {ts:.6f}")

            os.makedirs(
                "output",
                exist_ok=True
            )

            filename = "output/camera_test.jpg"

            cv2.imwrite(
                filename,
                frame
            )

            print(f"Image saved   : {filename}")

        cam.release()

    except Exception as e:

        print(e)
