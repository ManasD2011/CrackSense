from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import threading
import math

app = Flask(__name__)


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 640
HEIGHT = 480

# Actual ArUco marker side length in meters
# Example: 10 cm marker = 0.10
MARKER_SIZE = 0.059


# ============================================================
# CAMERA CALIBRATION
# ============================================================
#
# REPLACE THESE WITH YOUR IMX219 CALIBRATION VALUES
#

FX = 890.241265
FY = 887.723614
CX = 377.331918
CY = 200.943233

camera_matrix = np.array([
    [FX, 0,  CX],
    [0,  FY, CY],
    [0,  0,   1]
], dtype=np.float32)


# Replace with your actual distortion coefficients
dist_coeffs = np.array([
    -0.36967095,
    0.3668438,
    0.00159773,
    0.00323534,
    -0.6660792
], dtype=np.float32)


# ============================================================
# CAMERA
# ============================================================

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
        "size": (WIDTH, HEIGHT),
        "format": "RGB888"
    },
    controls={
        "FrameRate": 30
    }
)

picam2.configure(config)
picam2.start()

time.sleep(2)


# ============================================================
# ARUCO
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    parameters
)


# ============================================================
# SHARED DATA
# ============================================================

latest_frame = None
latest_data = []

frame_lock = threading.Lock()

running = True

fps = 0.0


# ============================================================
# ROTATION MATRIX -> EULER ANGLES
# ============================================================

def rotation_matrix_to_euler(R):

    sy = math.sqrt(
        R[0, 0] * R[0, 0] +
        R[1, 0] * R[1, 0]
    )

    singular = sy < 1e-6

    if not singular:

        roll = math.atan2(
            R[2, 1],
            R[2, 2]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        yaw = math.atan2(
            R[1, 0],
            R[0, 0]
        )

    else:

        roll = math.atan2(
            -R[1, 2],
            R[1, 1]
        )

        pitch = math.atan2(
            -R[2, 0],
            sy
        )

        yaw = 0

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw)
    )


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global latest_frame
    global latest_data
    global fps

    last_detection = 0

    last_time = time.perf_counter()

    frame_count = 0

    while running:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        frame = picam2.capture_array()
        
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        current_time = time.perf_counter()

        data = []

        # ----------------------------------------------------
        # ArUco detection
        # ----------------------------------------------------

        if current_time - last_detection >= 1.0 / 60:

            last_detection = current_time

            corners, ids, rejected = detector.detectMarkers(
                gray
            )

            if ids is not None:

                cv2.aruco.drawDetectedMarkers(
                    frame,
                    corners,
                    ids
                )

                for i, marker_id in enumerate(ids):

                    marker_id = int(marker_id)

                    pts = corners[i][0]

                    # ==================================================
                    # PIXEL CENTER
                    # ==================================================

                    pixel_x = int(
                        np.mean(pts[:, 0])
                    )

                    pixel_y = int(
                        np.mean(pts[:, 1])
                    )

                    # Image center
                    image_cx = WIDTH / 2
                    image_cy = HEIGHT / 2

                    # Pixel deviation
                    pixel_error_x = (
                        pixel_x - image_cx
                    )

                    pixel_error_y = (
                        pixel_y - image_cy
                    )


                    # ==================================================
                    # SOLVE PNP
                    # ==================================================

                    half = MARKER_SIZE / 2

                    object_points = np.array([
                        [-half,  half, 0],
                        [ half,  half, 0],
                        [ half, -half, 0],
                        [-half, -half, 0]
                    ], dtype=np.float32)


                    image_points = np.array(
                        pts,
                        dtype=np.float32
                    )


                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )


                    if success:

                        # ==================================================
                        # POSITION
                        # ==================================================

                        X = float(tvec[0][0])
                        Y = float(tvec[1][0])
                        Z = float(tvec[2][0])


                        # ==================================================
                        # ROTATION
                        # ==================================================

                        R, _ = cv2.Rodrigues(
                            rvec
                        )

                        roll, pitch, yaw = (
                            rotation_matrix_to_euler(R)
                        )


                        # ==================================================
                        # DRAW 3D AXES
                        # ==================================================

                        axis_length = MARKER_SIZE * 0.7

                        axis_points = np.float32([
                            [0, 0, 0],
                            [axis_length, 0, 0],
                            [0, axis_length, 0],
                            [0, 0, -axis_length]
                        ])


                        projected, _ = cv2.projectPoints(
                            axis_points,
                            rvec,
                            tvec,
                            camera_matrix,
                            dist_coeffs
                        )

                        projected = projected.reshape(-1, 2)


                        origin = tuple(
                            projected[0].astype(int)
                        )

                        x_axis = tuple(
                            projected[1].astype(int)
                        )

                        y_axis = tuple(
                            projected[2].astype(int)
                        )

                        z_axis = tuple(
                            projected[3].astype(int)
                        )


                        # X axis
                        cv2.line(
                            frame,
                            origin,
                            x_axis,
                            (255, 0, 0),
                            3
                        )

                        # Y axis
                        cv2.line(
                            frame,
                            origin,
                            y_axis,
                            (0, 255, 0),
                            3
                        )

                        # Z axis
                        cv2.line(
                            frame,
                            origin,
                            z_axis,
                            (0, 0, 255),
                            3
                        )


                        # ==================================================
                        # DISPLAY TEXT
                        # ==================================================

                        cv2.putText(
                            frame,
                            f"ID: {marker_id}",
                            (pixel_x + 15, pixel_y - 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2
                        )


                        cv2.putText(
                            frame,
                            f"X:{X:.3f}m Y:{Y:.3f}m Z:{Z:.3f}m",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2
                        )


                        cv2.putText(
                            frame,
                            f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}",
                            (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2
                        )


                        cv2.putText(
                            frame,
                            f"PX:{pixel_error_x:.1f} PY:{pixel_error_y:.1f}",
                            (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 0),
                            2
                        )


                        # ==================================================
                        # DATA FOR WEB PAGE
                        # ==================================================

                        data.append({

                            "id": marker_id,

                            "pixel_x": pixel_x,
                            "pixel_y": pixel_y,

                            "pixel_error_x":
                                pixel_error_x,

                            "pixel_error_y":
                                pixel_error_y,

                            "x": X,
                            "y": Y,
                            "z": Z,

                            "yaw": yaw,
                            "pitch": pitch,
                            "roll": roll

                        })


        # ====================================================
        # IMAGE CENTER
        # ====================================================

        image_cx = WIDTH // 2
        image_cy = HEIGHT // 2

        cv2.circle(
            frame,
            (image_cx, image_cy),
            5,
            (0, 0, 255),
            -1
        )

        cv2.line(
            frame,
            (image_cx - 20, image_cy),
            (image_cx + 20, image_cy),
            (0, 0, 255),
            1
        )

        cv2.line(
            frame,
            (image_cx, image_cy - 20),
            (image_cx, image_cy + 20),
            (0, 0, 255),
            1
        )


        # ====================================================
        # FPS
        # ====================================================

        frame_count += 1

        now = time.perf_counter()

        if now - last_time >= 1.0:

            fps = (
                frame_count /
                (now - last_time)
            )

            frame_count = 0

            last_time = now


        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ====================================================
        # JPEG
        # ====================================================

        # ====================================================
        # ROTATE ONLY FOR WEB DISPLAY
        # ====================================================

        display_frame = cv2.rotate(
            frame,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

        frame_bgr = cv2.cvtColor(
            display_frame,
            cv2.COLOR_RGB2BGR
        )

        success, jpeg = cv2.imencode(
            ".jpg",
            frame_bgr,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                70
            ]
        )

        if success:

            with frame_lock:

                latest_frame = jpeg.tobytes()

                latest_data = data


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    last_sent = None

    while True:

        with frame_lock:

            frame = latest_frame

        if frame is None:

            time.sleep(0.01)

            continue

        if frame == last_sent:

            time.sleep(0.005)

            continue

        last_sent = frame

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )


# ============================================================
# WEB PAGE
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>Pi 5 ArUco Pose</title>

<style>

body {
    background: #111;
    color: white;
    font-family: Arial;
    margin: 20px;
}

.container {
    max-width: 1000px;
    margin: auto;
}

h1 {
    text-align: center;
}

.video {
    width: 640px;
    max-width: 100%;
    display: block;
    margin: auto;
}

.panel {
    background: #222;
    padding: 20px;
    margin-top: 20px;
    border-radius: 10px;
}

.data {
    font-family: monospace;
    font-size: 17px;
    line-height: 1.8;
}

</style>

</head>

<body>

<div class="container">

<h1>Raspberry Pi 5 ArUco Pose</h1>

<img
    class="video"
    src="/video_feed"
>

<div class="panel">

<h2>Live Pose Data</h2>

<div id="data" class="data">
Loading...
</div>

</div>

</div>


<script>

async function updateData() {

    try {

        const response =
            await fetch("/data");

        const data =
            await response.json();

        let html = "";

        html +=
            "FPS: "
            + data.fps.toFixed(1)
            + "<br><br>";


        if (data.markers.length === 0) {

            html +=
                "<span style='color:red'>" +
                "NO ARUCO MARKER" +
                "</span>";

        }

        else {

            data.markers.forEach(
                function(m) {

                    html +=
                        "<b>MARKER ID: "
                        + m.id
                        + "</b><br><br>";

                    html +=
                        "Pixel Position: ("
                        + m.pixel_x
                        + ", "
                        + m.pixel_y
                        + ")<br>";

                    html +=
                        "Pixel Error X: "
                        + m.pixel_error_x.toFixed(2)
                        + " px<br>";

                    html +=
                        "Pixel Error Y: "
                        + m.pixel_error_y.toFixed(2)
                        + " px<br><br>";


                    html +=
                        "<b>3D POSITION</b><br>";

                    html +=
                        "X: "
                        + m.x.toFixed(3)
                        + " m<br>";

                    html +=
                        "Y: "
                        + m.y.toFixed(3)
                        + " m<br>";

                    html +=
                        "Z: "
                        + m.z.toFixed(3)
                        + " m<br><br>";


                    html +=
                        "<b>ORIENTATION</b><br>";

                    html +=
                        "Yaw: "
                        + m.yaw.toFixed(2)
                        + " deg<br>";

                    html +=
                        "Pitch: "
                        + m.pitch.toFixed(2)
                        + " deg<br>";

                    html +=
                        "Roll: "
                        + m.roll.toFixed(2)
                        + " deg<br>";

                }
            );

        }

        document.getElementById(
            "data"
        ).innerHTML = html;

    }

    catch(error) {

        console.log(error);

    }

}


setInterval(
    updateData,
    100
);

updateData();

</script>

</body>

</html>

"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/data")
def data():

    with frame_lock:

        markers = latest_data.copy()

    return {

        "fps": fps,

        "markers": markers

    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    thread.start()

    print()
    print("==============================")
    print(" Raspberry Pi 5 ArUco Pose")
    print("==============================")
    print()
    print("Open on your laptop:")
    print()
    print("http://<PI_IP>:5000")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True
    )
