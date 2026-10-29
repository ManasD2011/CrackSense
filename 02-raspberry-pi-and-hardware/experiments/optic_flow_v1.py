from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)

# ============================================================
# CAMERA
# ============================================================

WIDTH = 640
HEIGHT = 480

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
        "size": (WIDTH, HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

time.sleep(2)


# ============================================================
# OPTICAL FLOW PARAMETERS
# ============================================================

feature_params = dict(
    maxCorners=300,
    qualityLevel=0.003,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)


# ============================================================
# SHARED FRAME
# ============================================================

latest_frame = None

frame_lock = threading.Lock()

running = True


# ============================================================
# OPTICAL FLOW PROCESSING
# ============================================================

def camera_loop():

    global latest_frame

    frame = picam2.capture_array()

    old_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY
    )

    p0 = cv2.goodFeaturesToTrack(
        old_gray,
        mask=None,
        **feature_params
    )

    frame_count = 0
    fps_start = time.perf_counter()

    while running:

        # ----------------------------------------------------
        # Capture
        # ----------------------------------------------------

        frame = picam2.capture_array()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )


        # ----------------------------------------------------
        # Re-detect features if necessary
        # ----------------------------------------------------

        if p0 is None or len(p0) < 20:

            p0 = cv2.goodFeaturesToTrack(
                old_gray,
                mask=None,
                **feature_params
            )

            old_gray = gray.copy()

            continue


        # ----------------------------------------------------
        # Lucas-Kanade optical flow
        # ----------------------------------------------------

        p1, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            gray,
            p0,
            None,
            **lk_params
        )


        if p1 is None:

            old_gray = gray.copy()

            p0 = cv2.goodFeaturesToTrack(
                old_gray,
                mask=None,
                **feature_params
            )

            continue


        # ----------------------------------------------------
        # Keep valid features
        # ----------------------------------------------------

        good_new = p1[status == 1]
        good_old = p0[status == 1]


        if len(good_new) == 0:

            old_gray = gray.copy()

            p0 = cv2.goodFeaturesToTrack(
                old_gray,
                mask=None,
                **feature_params
            )

            continue


        # ----------------------------------------------------
        # Calculate displacement
        # ----------------------------------------------------

        displacement = good_new - good_old

        dx = displacement[:, 0]
        dy = displacement[:, 1]


        median_dx = float(np.median(dx))
        median_dy = float(np.median(dy))

        median_motion = float(
            np.median(
                np.sqrt(dx**2 + dy**2)
            )
        )


        # ----------------------------------------------------
        # Create display image
        # ----------------------------------------------------

        display = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )


        # ----------------------------------------------------
        # Draw optical-flow vectors
        # ----------------------------------------------------

        for new, old in zip(good_new, good_old):

            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()

            x_new = int(x_new)
            y_new = int(y_new)

            x_old = int(x_old)
            y_old = int(y_old)

            # Motion vector
            cv2.arrowedLine(
                display,
                (x_old, y_old),
                (x_new, y_new),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

            # Feature point
            cv2.circle(
                display,
                (x_new, y_new),
                3,
                (255, 0, 0),
                -1
            )


        # ----------------------------------------------------
        # Information overlay
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (0, 0),
            (310, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display,
            f"Features: {len(good_new)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"dx: {median_dx:.2f} px/frame",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"dy: {median_dy:.2f} px/frame",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Motion: {median_motion:.2f} px/frame",
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # FPS measurement
        # ----------------------------------------------------

        frame_count += 1

        elapsed = time.perf_counter() - fps_start

        if elapsed >= 2.0:

            actual_fps = frame_count / elapsed

            print(
                f"FPS: {actual_fps:.2f} | "
                f"Features: {len(good_new)} | "
                f"dx: {median_dx:.2f} | "
                f"dy: {median_dy:.2f} | "
                f"motion: {median_motion:.2f}"
            )

            frame_count = 0
            fps_start = time.perf_counter()


        # ----------------------------------------------------
        # JPEG encode
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            display,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if success:

            with frame_lock:

                latest_frame = buffer.tobytes()


        # ----------------------------------------------------
        # Next iteration
        # ----------------------------------------------------

        old_gray = gray.copy()

        p0 = good_new.reshape(-1, 1, 2)


# ============================================================
# WEB STREAM
# ============================================================

def generate_frames():

    while True:

        with frame_lock:

            frame = latest_frame

        if frame is None:

            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ============================================================
# WEB PAGE
# ============================================================

@app.route("/")
def index():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>Raspberry Pi Optical Flow</title>

        <style>

            body {
                background: #111;
                color: white;
                font-family: Arial;
                text-align: center;
            }

            img {
                width: 640px;
                max-width: 95%;
                height: auto;
            }

        </style>

    </head>

    <body>

        <h2>IMX219 Optical Flow</h2>

        <img src="/video">

        <p>
        Green arrows = optical flow<br>
        Blue dots = tracked features
        </p>

    </body>

    </html>
    """


# ============================================================
# VIDEO ENDPOINT
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    thread.start()

    print()
    print("======================================")
    print(" Raspberry Pi Optical Flow Server")
    print("======================================")
    print()
    print("Open on laptop:")
    print("http://10.96.141.185:8080")
    print()

    app.run(
        host="0.0.0.0",
        port=8080,
        threaded=True
    )
