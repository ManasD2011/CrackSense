"""
optical_flow.py

Elementary Optical Flow Module

Stage 1:
    - Initialize optical flow
    - Detect Shi-Tomasi features

Author: Shubham
"""

import cv2
import numpy as np
import time


class OpticalFlow:

    def __init__(self):

        # ---------------- Feature Detector ----------------

        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )

        # ---------------- Lucas-Kanade ----------------

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                30,
                0.01
            )
        )

        # ---------------- Variables ----------------

        self.initialized = False

        self.old_gray = None
        self.p0 = None

        self.mask = None

        self.prev_time = time.time()

    # =====================================================

    def initialize(self, frame):

        """
        Initialize optical flow using first frame.
        """

        self.old_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        self.p0 = cv2.goodFeaturesToTrack(
            self.old_gray,
            mask=None,
            **self.feature_params
        )

        self.mask = np.zeros_like(frame)

        self.initialized = True

        print("\n========== OPTICAL FLOW ==========\n")

        if self.p0 is None:

            print("Initialization Failed")
            print("No Features Detected")

            return

        print("Initialization Successful")
        print(f"Detected Features : {len(self.p0)}")

    # =====================================================

    def process(self, frame):
                """
                Process one frame and estimate optical flow.
                """

                # --------------------------------------------------
                # First Frame
                # --------------------------------------------------

                if not self.initialized:

                    self.initialize(frame)

                    return None

                # --------------------------------------------------
                # Convert Current Frame
                # --------------------------------------------------

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                # --------------------------------------------------
                # Re-detect Features if Too Few
                # --------------------------------------------------

                if self.p0 is None or len(self.p0) < 150:

                    self.p0 = cv2.goodFeaturesToTrack(
                        gray,
                        mask=None,
                        **self.feature_params
                    )

                    self.old_gray = gray.copy()

                    if self.p0 is None:

                        print("No Features Found")

                        return None

                    print(f"Reinitialized Features : {len(self.p0)}")

                    return None

                # --------------------------------------------------
                # Lucas-Kanade Tracking
                # --------------------------------------------------

                p1, st, err = cv2.calcOpticalFlowPyrLK(

                    self.old_gray,
                    gray,
                    self.p0,
                    None,
                    **self.lk_params

                )

                if p1 is None:

                    self.initialized = False

                    return None

                # --------------------------------------------------
                # Remove Invalid Tracks
                # --------------------------------------------------

                st = st.reshape(-1)
                err = err.reshape(-1)

                valid = (st == 1) & (err < 20)

                good_new = p1[valid]
                good_old = self.p0[valid]

                if len(good_new) == 0:

                    self.initialized = False

                    return None

                # --------------------------------------------------
                # Compute Flow Vectors
                # --------------------------------------------------

                flow = good_new - good_old

                lengths = np.linalg.norm(flow, axis=1)

                motion_mask = lengths < 25

                good_new = good_new[motion_mask]
                good_old = good_old[motion_mask]

                if len(good_new) == 0:

                    self.initialized = False

                    return None

                flow = good_new - good_old

                # --------------------------------------------------
                # Robust Motion Estimate
                # --------------------------------------------------

                dx = float(np.median(flow[:, 0]))
                dy = float(np.median(flow[:, 1]))

                # --------------------------------------------------
                # FPS
                # --------------------------------------------------

                current_time = time.time()

                fps = 1.0 / (current_time - self.prev_time)

                self.prev_time = current_time

                # --------------------------------------------------
                # Update State
                # --------------------------------------------------

                self.old_gray = gray.copy()

                self.p0 = good_new.reshape(-1, 1, 2)

                # --------------------------------------------------
                # Return Result
                # --------------------------------------------------

                return {

                    "dx": dx,

                    "dy": dy,

                    "tracked": len(good_new),

                    "fps": fps,

                    "good_new": good_new,

                    "good_old": good_old

                }


# =========================================================
# Standalone Test
# =========================================================

if __name__ == "__main__":

    from camera import Camera

    cam = Camera()

    flow = OpticalFlow()

    print("\n===== Optical Flow Test =====\n")

    while True:

        frame, timestamp = cam.read()

        result = flow.process(frame)

        if result is None:
            continue

        print(
            f"Tracked: {result['tracked']:3d} | "
            f"dx: {result['dx']:7.2f} | "
            f"dy: {result['dy']:7.2f} | "
            f"FPS: {result['fps']:5.1f}"
        )

        # Press Ctrl+C to stop
