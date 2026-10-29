import cv2
import numpy as np
import time
import threading
import math
import subprocess
import os
import atexit

from flask import Flask, Response, render_template, jsonify
from picamera2 import Picamera2
from pymavlink import mavutil


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# SETTINGS
# ============================================================

# ------------------------------------------------------------
# IMX219
# ------------------------------------------------------------

IMX_WIDTH = 640
IMX_HEIGHT = 480
IMX_FPS = 30


# ------------------------------------------------------------
# ArUco
# ------------------------------------------------------------

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50

# Actual physical marker size
# Your previous code uses 0.059 m = 5.9 cm
MARKER_SIZE = 0.059


# ------------------------------------------------------------
# USB WEBCAM
# ------------------------------------------------------------

USB_CAMERA = "/dev/video8"

USB_WIDTH = 1280
USB_HEIGHT = 720
USB_FPS = 30


# ------------------------------------------------------------
# Webcam recording
# ------------------------------------------------------------

RECORDINGS_DIR = os.path.expanduser(
    "~/drone_dashboard/recordings"
)

# Each recording file = 5 minutes
RECORDING_SEGMENT_SECONDS = 300


# ------------------------------------------------------------
# Pixhawk / LiDAR
# ------------------------------------------------------------

PIXHAWK_PORT = "/dev/ttyACM0"
PIXHAWK_BAUD = 115200

LIDAR_RATE_HZ = 10


# ============================================================
# CREATE RECORDINGS DIRECTORY
# ============================================================

os.makedirs(
    RECORDINGS_DIR,
    exist_ok=True
)


# ============================================================
# CAMERA CALIBRATION
# ============================================================
#
# These are the values from your earlier ArUco code.
#
# Replace them only if you have newer calibration values.
#

FX = 890.241265
FY = 887.723614
CX = 377.331918
CY = 200.943233


camera_matrix = np.array(
    [
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1]
    ],
    dtype=np.float32
)


dist_coeffs = np.array(
    [
        -0.36967095,
        0.3668438,
        0.00159773,
        0.00323534,
        -0.6660792
    ],
    dtype=np.float32
)


# ============================================================
# GLOBAL DATA
# ============================================================

# ------------------------------------------------------------
# IMX219
# ------------------------------------------------------------

imx_jpeg = None

imx_data = []

imx_lock = threading.Lock()


# ------------------------------------------------------------
# USB webcam
# ------------------------------------------------------------

usb_jpeg = None

usb_lock = threading.Lock()


# ------------------------------------------------------------
# LiDAR
# ------------------------------------------------------------

lidar_distance_cm = None

lidar_lock = threading.Lock()


# ------------------------------------------------------------
# FPS
# ------------------------------------------------------------

imx_fps = 0.0
usb_fps = 0.0


# ------------------------------------------------------------
# FFmpeg process
# ------------------------------------------------------------

ffmpeg_process = None


# ============================================================
# IMX219 INITIALIZATION
# ============================================================

print()
print("============================================")
print("Initializing IMX219")
print("============================================")


picam2 = Picamera2()


imx_config = picam2.create_video_configuration(

    main={
        "size": (
            IMX_WIDTH,
            IMX_HEIGHT
        ),

        "format": "RGB888"
    },

    controls={
        "FrameRate": IMX_FPS
    }
)


picam2.configure(
    imx_config
)


picam2.start()


time.sleep(2)


print(
    f"IMX219 started: "
    f"{IMX_WIDTH}x{IMX_HEIGHT} @ {IMX_FPS} FPS"
)


# ============================================================
# ARUCO INITIALIZATION
# ============================================================

print()
print("============================================")
print("Initializing ArUco")
print("============================================")


aruco_dict = cv2.aruco.getPredefinedDictionary(
    ARUCO_DICTIONARY
)


parameters = cv2.aruco.DetectorParameters()


detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    parameters
)


print(
    "Dictionary: DICT_4X4_50"
)


print(
    f"Marker size: {MARKER_SIZE} m"
)


# ============================================================
# ROTATION MATRIX -> EULER ANGLES
# ============================================================

def rotation_matrix_to_euler(R):

    sy = math.sqrt(

        R[0, 0] * R[0, 0]
        +
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
# IMX219 CAMERA + ARUCO LOOP
# ============================================================

def imx219_loop():

    global imx_jpeg
    global imx_data
    global imx_fps


    print(
        "IMX219 + ArUco thread started"
    )


    frame_count = 0

    fps_start = time.perf_counter()


    while True:

        try:

            # =================================================
            # CAPTURE
            # =================================================

            frame = picam2.capture_array()


            # =================================================
            # RGB -> GRAYSCALE
            # =================================================

            gray = cv2.cvtColor(

                frame,

                cv2.COLOR_RGB2GRAY

            )


            # =================================================
            # ARUCO DETECTION
            # =================================================

            corners, ids, rejected = (
                detector.detectMarkers(gray)
            )


            data = []


            # =================================================
            # IMAGE CENTER
            # =================================================

            image_cx = IMX_WIDTH / 2.0
            image_cy = IMX_HEIGHT / 2.0


            # =================================================
            # ARUCO MARKERS FOUND
            # =================================================

            if ids is not None:

                cv2.aruco.drawDetectedMarkers(

                    frame,

                    corners,

                    ids

                )


                # ---------------------------------------------
                # Process every detected marker
                # ---------------------------------------------

                for i, marker_id in enumerate(ids):


                    marker_id = int(
                        marker_id
                    )


                    pts = corners[i][0]


                    # =========================================
                    # PIXEL CENTER
                    # =========================================

                    pixel_x = int(
                        np.mean(
                            pts[:, 0]
                        )
                    )


                    pixel_y = int(
                        np.mean(
                            pts[:, 1]
                        )
                    )


                    # =========================================
                    # PIXEL ERROR
                    # =========================================

                    pixel_error_x = (
                        pixel_x - image_cx
                    )


                    pixel_error_y = (
                        pixel_y - image_cy
                    )


                    # =========================================
                    # OBJECT POINTS
                    # =========================================

                    half = (
                        MARKER_SIZE / 2.0
                    )


                    object_points = np.array(

                        [

                            [-half, half, 0],

                            [half, half, 0],

                            [half, -half, 0],

                            [-half, -half, 0]

                        ],

                        dtype=np.float32

                    )


                    image_points = np.array(

                        pts,

                        dtype=np.float32

                    )


                    # =========================================
                    # SOLVE PNP
                    # =========================================

                    success, rvec, tvec = (
                        cv2.solvePnP(

                            object_points,

                            image_points,

                            camera_matrix,

                            dist_coeffs,

                            flags=(
                                cv2.SOLVEPNP_IPPE_SQUARE
                            )

                        )
                    )


                    if not success:

                        continue


                    # =========================================
                    # POSITION
                    # =========================================

                    X = float(
                        tvec[0][0]
                    )


                    Y = float(
                        tvec[1][0]
                    )


                    Z = float(
                        tvec[2][0]
                    )


                    # =========================================
                    # ROTATION
                    # =========================================

                    R, _ = cv2.Rodrigues(
                        rvec
                    )


                    roll, pitch, yaw = (
                        rotation_matrix_to_euler(
                            R
                        )
                    )


                    # =========================================
                    # DRAW 3D AXES
                    # =========================================

                    axis_length = (
                        MARKER_SIZE * 0.7
                    )


                    axis_points = np.float32(

                        [

                            [0, 0, 0],

                            [axis_length, 0, 0],

                            [0, axis_length, 0],

                            [0, 0, -axis_length]

                        ]

                    )


                    projected, _ = (
                        cv2.projectPoints(

                            axis_points,

                            rvec,

                            tvec,

                            camera_matrix,

                            dist_coeffs

                        )
                    )


                    projected = (
                        projected.reshape(
                            -1,
                            2
                        )
                    )


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


                    # =========================================
                    # TEXT
                    # =========================================

                    cv2.putText(

                        frame,

                        f"ID: {marker_id}",

                        (
                            pixel_x + 15,
                            pixel_y - 65
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.55,

                        (0, 255, 0),

                        2

                    )


                    cv2.putText(

                        frame,

                        (
                            f"X:{X:.3f}m "
                            f"Y:{Y:.3f}m "
                            f"Z:{Z:.3f}m"
                        ),

                        (10, 60),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.55,

                        (255, 255, 255),

                        2

                    )


                    cv2.putText(

                        frame,

                        (
                            f"Yaw:{yaw:.1f} "
                            f"Pitch:{pitch:.1f} "
                            f"Roll:{roll:.1f}"
                        ),

                        (10, 85),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.55,

                        (255, 255, 255),

                        2

                    )


                    cv2.putText(

                        frame,

                        (
                            f"PX:{pixel_error_x:.1f} "
                            f"PY:{pixel_error_y:.1f}"
                        ),

                        (10, 110),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.55,

                        (255, 255, 0),

                        2

                    )


                    # =========================================
                    # STORE DATA
                    # =========================================

                    data.append({

                        "id": marker_id,

                        "pixel_x": pixel_x,

                        "pixel_y": pixel_y,

                        "pixel_error_x":
                            float(pixel_error_x),

                        "pixel_error_y":
                            float(pixel_error_y),

                        "x": X,

                        "y": Y,

                        "z": Z,

                        "yaw": yaw,

                        "pitch": pitch,

                        "roll": roll

                    })


            # =================================================
            # IMAGE CENTER
            # =================================================

            image_cx_int = (
                IMX_WIDTH // 2
            )


            image_cy_int = (
                IMX_HEIGHT // 2
            )


            cv2.circle(

                frame,

                (
                    image_cx_int,
                    image_cy_int
                ),

                5,

                (0, 0, 255),

                -1

            )


            cv2.line(

                frame,

                (
                    image_cx_int - 20,
                    image_cy_int
                ),

                (
                    image_cx_int + 20,
                    image_cy_int
                ),

                (0, 0, 255),

                1

            )


            cv2.line(

                frame,

                (
                    image_cx_int,
                    image_cy_int - 20
                ),

                (
                    image_cx_int,
                    image_cy_int + 20
                ),

                (0, 0, 255),

                1

            )


            # =================================================
            # FPS
            # =================================================

            frame_count += 1


            now = time.perf_counter()


            if (
                now - fps_start
            ) >= 1.0:

                imx_fps = (
                    frame_count /
                    (now - fps_start)
                )


                frame_count = 0

                fps_start = now


            cv2.putText(

                frame,

                f"FPS: {imx_fps:.1f}",

                (10, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2

            )


            # =================================================
            # RGB -> BGR
            # =================================================

            frame_bgr = cv2.cvtColor(

                frame,

                cv2.COLOR_RGB2BGR

            )


            # =================================================
            # JPEG
            # =================================================

            success, jpeg = cv2.imencode(

                ".jpg",

                frame_bgr,

                [

                    cv2.IMWRITE_JPEG_QUALITY,

                    70

                ]

            )


            if success:

                with imx_lock:

                    imx_jpeg = (
                        jpeg.tobytes()
                    )

                    imx_data = data


        except Exception as e:

            print(
                "IMX219/ArUco error:",
                e
            )

            time.sleep(0.1)


# ============================================================
# USB WEBCAM
# ============================================================

def usb_camera_thread():

    global usb_jpeg
    global usb_fps
    global ffmpeg_process


    print()
    print("============================================")
    print("Initializing USB Webcam")
    print("============================================")


    print(
        f"Device     : {USB_CAMERA}"
    )


    print(
        f"Resolution : "
        f"{USB_WIDTH}x{USB_HEIGHT}"
    )


    print(
        f"FPS        : {USB_FPS}"
    )


    print(
        "Format     : MJPEG"
    )


    print()


    # ========================================================
    # CONFIGURE CAMERA
    # ========================================================

    configure_command = [

        "v4l2-ctl",

        "-d",
        USB_CAMERA,

        "--set-fmt-video="
        f"width={USB_WIDTH},"
        f"height={USB_HEIGHT},"
        f"pixelformat=MJPG",

        "--set-parm",
        str(USB_FPS)

    ]


    result = subprocess.run(

        configure_command,

        capture_output=True,

        text=True

    )


    if result.returncode != 0:

        print(
            "ERROR configuring USB webcam:"
        )

        print(
            result.stderr
        )

        return


    print(
        "USB webcam configured successfully"
    )


    # ========================================================
    # ONE FFMPEG PROCESS
    #
    # Output 1:
    #     MJPEG -> Python -> browser
    #
    # Output 2:
    #     MJPEG -> AVI recording
    #
    # No decode/re-encode of webcam video.
    # ========================================================

    recording_pattern = os.path.join(

        RECORDINGS_DIR,

        "webcam_%Y%m%d_%H%M%S.avi"

    )


    ffmpeg_command = [

        "ffmpeg",

        "-hide_banner",

        "-loglevel",
        "warning",

        "-f",
        "v4l2",

        "-input_format",
        "mjpeg",

        "-video_size",
        f"{USB_WIDTH}x{USB_HEIGHT}",

        "-framerate",
        str(USB_FPS),

        "-i",
        USB_CAMERA,


        # ====================================================
        # OUTPUT 1 - WEB STREAM
        # ====================================================

        "-map",
        "0:v",

        "-c:v",
        "copy",

        "-f",
        "mjpeg",

        "pipe:1",


        # ====================================================
        # OUTPUT 2 - RECORDING
        # ====================================================

        "-map",
        "0:v",

        "-c:v",
        "copy",

        "-f",
        "segment",

        "-segment_time",
        str(RECORDING_SEGMENT_SECONDS),

        "-reset_timestamps",
        "1",

        "-segment_format",
        "avi",

        "-strftime",
        "1",

        recording_pattern

    ]


    print()
    print(
        "Starting USB webcam stream + recording..."
    )


    print(
        "Recording directory:"
    )

    print(
        RECORDINGS_DIR
    )

    print(
        f"Segment length: "
        f"{RECORDING_SEGMENT_SECONDS} seconds"
    )

    print()


    try:

        ffmpeg_process = subprocess.Popen(

            ffmpeg_command,

            stdout=subprocess.PIPE,

            stderr=subprocess.DEVNULL,

            bufsize=0

        )

    except Exception as e:

        print(
            "Could not start FFmpeg:"
        )

        print(e)

        return


    print(
        "USB webcam stream started"
    )


    print(
        "USB webcam recording started"
    )


    # ========================================================
    # READ MJPEG STREAM
    # ========================================================

    buffer = bytearray()


    frame_count = 0

    fps_start = time.perf_counter()


    while True:

        try:

            chunk = (
                ffmpeg_process.stdout.read(
                    16384
                )
            )


            if not chunk:

                print(
                    "USB webcam stream stopped"
                )

                break


            buffer.extend(
                chunk
            )


            # =================================================
            # FIND JPEG FRAMES
            # =================================================

            while True:

                # JPEG Start Of Image
                start = buffer.find(
                    b"\xff\xd8"
                )


                if start == -1:

                    # Prevent unlimited buffer
                    if len(buffer) > 2_000_000:

                        buffer.clear()

                    break


                # Remove bytes before JPEG
                if start > 0:

                    del buffer[:start]


                # JPEG End Of Image
                end = buffer.find(

                    b"\xff\xd9",

                    2

                )


                if end == -1:

                    break


                end += 2


                jpeg = bytes(
                    buffer[:end]
                )


                del buffer[:end]


                # =================================================
                # STORE LATEST FRAME
                # =================================================

                with usb_lock:

                    usb_jpeg = jpeg


                # =================================================
                # FPS
                # =================================================

                frame_count += 1


                now = time.perf_counter()


                if (
                    now - fps_start
                ) >= 1.0:

                    usb_fps = (
                        frame_count /
                        (now - fps_start)
                    )


                    frame_count = 0

                    fps_start = now


        except Exception as e:

            print(
                "USB webcam error:",
                e
            )

            break


# ============================================================
# USB WEBCAM STREAM
# ============================================================

def usb_stream():

    last_frame = None


    while True:

        with usb_lock:

            frame = usb_jpeg


        if frame is None:

            time.sleep(
                0.005
            )

            continue


        # Avoid sending same frame repeatedly

        if frame == last_frame:

            time.sleep(
                0.001
            )

            continue


        last_frame = frame


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n"

            b"Content-Length: "
            + str(len(frame)).encode()
            + b"\r\n\r\n"

            + frame

            + b"\r\n"

        )


# ============================================================
# IMX219 STREAM
# ============================================================

def imx_stream():

    last_frame = None


    while True:

        with imx_lock:

            frame = imx_jpeg

            if frame is not None:

                frame = bytes(
                    frame
                )


        if frame is None:

            time.sleep(
                0.01
            )

            continue


        if frame == last_frame:

            time.sleep(
                0.005
            )

            continue


        last_frame = frame


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n"

            b"Content-Length: "
            + str(len(frame)).encode()
            + b"\r\n\r\n"

            + frame

            + b"\r\n"

        )


# ============================================================
# LIDAR THREAD
# ============================================================

def lidar_thread():

    global lidar_distance_cm


    print()
    print("============================================")
    print("Connecting to Pixhawk")
    print("============================================")


    print(
        f"Port: {PIXHAWK_PORT}"
    )


    print(
        f"Baud: {PIXHAWK_BAUD}"
    )


    try:

        connection = (
            mavutil.mavlink_connection(

                PIXHAWK_PORT,

                baud=PIXHAWK_BAUD

            )
        )


        print(
            "Waiting for heartbeat..."
        )


        connection.wait_heartbeat()


        print()
        print(
            "Pixhawk connected!"
        )


        print(
            f"System ID: "
            f"{connection.target_system}"
        )


        print(
            f"Component ID: "
            f"{connection.target_component}"
        )


        print(
            "----------------------------------------"
        )


        # ====================================================
        # DISTANCE_SENSOR @ 10 Hz
        # ====================================================

        connection.mav.command_long_send(

            connection.target_system,

            connection.target_component,

            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,

            0,

            132,

            100000,

            0,
            0,
            0,
            0,
            0

        )


        print(
            "Requested DISTANCE_SENSOR at 10 Hz"
        )


        print(
            "----------------------------------------"
        )


        while True:

            msg = connection.recv_match(

                type="DISTANCE_SENSOR",

                blocking=True,

                timeout=2

            )


            if msg is None:

                print(
                    "No DISTANCE_SENSOR received"
                )

                continue


            distance_cm = (
                msg.current_distance
            )


            with lidar_lock:

                lidar_distance_cm = (
                    distance_cm
                )


    except Exception as e:

        print()
        print(
            "Pixhawk/LiDAR ERROR:"
        )

        print(e)

        print()


# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    global ffmpeg_process


    print()
    print(
        "Stopping system..."
    )


    if ffmpeg_process is not None:

        try:

            ffmpeg_process.terminate()

            ffmpeg_process.wait(
                timeout=3
            )

        except Exception:

            try:

                ffmpeg_process.kill()

            except Exception:

                pass


    try:

        picam2.stop()

    except Exception:

        pass


    print(
        "System stopped."
    )


atexit.register(
    cleanup
)


# ============================================================
# WEB PAGE
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width,
             initial-scale=1.0"
>

<title>
Drone Sensor Dashboard
</title>


<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    background: #111;

    color: white;

    font-family: Arial, sans-serif;

}


header {

    background: #1b1b1b;

    padding: 15px;

    text-align: center;

    border-bottom: 1px solid #333;

}


header h1 {

    margin: 0;

    font-size: 24px;

}


.camera-grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 15px;

    padding: 15px;

}


.camera-card {

    background: #1b1b1b;

    padding: 10px;

    border-radius: 10px;

}


.camera-card h2 {

    font-size: 18px;

    margin: 5px 0 10px 5px;

}


.camera-card img {

    width: 100%;

    display: block;

    border-radius: 6px;

    background: black;

}


.info-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 15px;

    padding: 0 15px 15px 15px;

}


.card {

    background: #1b1b1b;

    padding: 20px;

    border-radius: 10px;

}


.card h2 {

    margin-top: 0;

}


.distance {

    font-size: 45px;

    font-weight: bold;

}


.data {

    font-family: monospace;

    font-size: 16px;

    line-height: 1.7;

}


.status {

    color: #aaa;

}


@media(max-width: 900px) {

    .camera-grid {

        grid-template-columns: 1fr;

    }

    .info-grid {

        grid-template-columns: 1fr;

    }

}

</style>

</head>


<body>


<header>

<h1>
Raspberry Pi 5 Drone Dashboard
</h1>

</header>


<!-- ======================================================
     CAMERAS
     ======================================================= -->

<div class="camera-grid">


<div class="camera-card">

<h2>
IMX219 — Downward Camera + ArUco
</h2>

<img
    src="/camera/imx"
    alt="IMX219 ArUco feed"
>

</div>


<div class="camera-card">

<h2>
USB Webcam — Recording Camera
</h2>

<img
    src="/camera/usb"
    alt="USB webcam feed"
>

</div>


</div>


<!-- ======================================================
     DATA
     ======================================================= -->

<div class="info-grid">


<div class="card">

<h2>
ArUco Data
</h2>

<div
    id="aruco-data"
    class="data"
>

Waiting...

</div>

</div>


<div class="card">

<h2>
1D LiDAR
</h2>


<div>

<span
    id="distance"
    class="distance"
>
---
</span>

<span>
m
</span>

</div>


<div
    id="lidar-status"
    class="status"
>

Waiting for LiDAR...

</div>

</div>


</div>


<script>


// ==========================================================
// UPDATE DASHBOARD DATA
// ==========================================================

async function updateData() {


    try {


        // ==================================================
        // ARUCO
        // ==================================================

        const arucoResponse =
            await fetch(
                "/api/aruco"
            );


        const arucoData =
            await arucoResponse.json();


        let html = "";


        html +=
            "Detection FPS: "
            + arucoData.fps.toFixed(1)
            + "<br><br>";


        if (
            arucoData.markers.length === 0
        ) {


            html +=
                "<span style='color:red'>" +
                "NO ARUCO MARKER" +
                "</span>";


        }

        else {


            arucoData.markers.forEach(

                function(m) {


                    html +=
                        "<b>MARKER ID: "
                        + m.id
                        + "</b><br>";


                    html +=
                        "Pixel: ("
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
                        "<b>POSITION</b><br>";


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
                        + "°<br>";


                    html +=
                        "Pitch: "
                        + m.pitch.toFixed(2)
                        + "°<br>";


                    html +=
                        "Roll: "
                        + m.roll.toFixed(2)
                        + "°<br><br>";

                }

            );

        }


        document.getElementById(
            "aruco-data"
        ).innerHTML = html;


        // ==================================================
        // LIDAR
        // ==================================================

        const lidarResponse =
            await fetch(
                "/api/lidar"
            );


        const lidarData =
            await lidarResponse.json();


        const distance =
            document.getElementById(
                "distance"
            );


        const status =
            document.getElementById(
                "lidar-status"
            );


        if (
            lidarData.distance_m !== null
        ) {


            distance.innerText =
                lidarData.distance_m.toFixed(2);


            status.innerText =
                "LiDAR data received";


        }

        else {


            distance.innerText =
                "---";


            status.innerText =
                "Waiting for LiDAR...";

        }


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

    return HTML


# ============================================================
# IMX219 ROUTE
# ============================================================

@app.route("/camera/imx")
def camera_imx():

    return Response(

        imx_stream(),

        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )

    )


# ============================================================
# USB CAMERA ROUTE
# ============================================================

@app.route("/camera/usb")
def camera_usb():

    return Response(

        usb_stream(),

        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )

    )


# ============================================================
# ARUCO API
# ============================================================

@app.route("/api/aruco")
def aruco_api():

    with imx_lock:

        markers = list(
            imx_data
        )


    return jsonify({

        "fps": imx_fps,

        "markers": markers

    })


# ============================================================
# LIDAR API
# ============================================================

@app.route("/api/lidar")
def lidar_api():

    with lidar_lock:

        distance = (
            lidar_distance_cm
        )


    if distance is None:

        return jsonify({

            "distance_cm": None,

            "distance_m": None

        })


    return jsonify({

        "distance_cm": distance,

        "distance_m":
            distance / 100.0

    })


# ============================================================
# START THREADS
# ============================================================

print()
print("============================================")
print("Starting threads")
print("============================================")


imx_thread = threading.Thread(

    target=imx219_loop,

    daemon=True

)


usb_thread = threading.Thread(

    target=usb_camera_thread,

    daemon=True

)


lidar_thread_obj = threading.Thread(

    target=lidar_thread,

    daemon=True

)


imx_thread.start()

usb_thread.start()

lidar_thread_obj.start()


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("============================================")
    print("       DRONE SENSOR WEB DASHBOARD")
    print("============================================")

    print()

    print(
        "IMX219 : /dev/video0"
    )

    print(
        "USB    : /dev/video8"
    )

    print(
        "LiDAR  : /dev/ttyACM0"
    )

    print()

    print(
        "ArUco  : DICT_4X4_50"
    )

    print(
        f"Marker : {MARKER_SIZE} m"
    )

    print()

    print(
        "USB recording:"
    )

    print(
        RECORDINGS_DIR
    )

    print(
        f"Segment: "
        f"{RECORDING_SEGMENT_SECONDS} seconds"
    )

    print()

    print(
        "Dashboard:"
    )

    print(
        "http://<RASPBERRY_PI_IP>:5000"
    )

    print()

    app.run(

        host="0.0.0.0",

        port=5000,

        threaded=True

    )