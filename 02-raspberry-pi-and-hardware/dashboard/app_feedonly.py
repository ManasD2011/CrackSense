import cv2
import time
import threading
import subprocess
import select

from flask import Flask, Response, render_template, jsonify
from picamera2 import Picamera2
from pymavlink import mavutil


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# IMX219
# ------------------------------------------------------------

IMX_WIDTH = 640
IMX_HEIGHT = 480
IMX_STREAM_FPS = 20


# ------------------------------------------------------------
# USB WEBCAM
# ------------------------------------------------------------

USB_CAMERA = "/dev/video8"

USB_WIDTH = 1280
USB_HEIGHT = 720
USB_FPS = 30


# ------------------------------------------------------------
# Pixhawk / LiDAR
# ------------------------------------------------------------

PIXHAWK_PORT = "/dev/ttyACM0"
PIXHAWK_BAUD = 115200

LIDAR_RATE_HZ = 10


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL DATA
# ============================================================

# IMX219
imx_frame = None
imx_lock = threading.Lock()


# USB webcam
usb_frame = None
usb_lock = threading.Lock()


# LiDAR
lidar_distance_cm = None
lidar_lock = threading.Lock()


# ============================================================
# IMX219 INITIALIZATION
# ============================================================

print()
print("============================================")
print("Initializing IMX219")
print("============================================")

picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={
        "size": (IMX_WIDTH, IMX_HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(camera_config)

picam2.start()

time.sleep(2)

print("IMX219 started")
print(
    f"Resolution: {IMX_WIDTH}x{IMX_HEIGHT}"
)


# ============================================================
# IMX219 CAPTURE
# ============================================================

def imx219_capture():

    global imx_frame

    print(
        "IMX219 capture thread started"
    )

    while True:

        try:

            frame = picam2.capture_array()

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            with imx_lock:

                imx_frame = frame

        except Exception as e:

            print(
                "IMX219 error:",
                e
            )

            time.sleep(0.1)


# ============================================================
# USB WEBCAM
#
# IMPORTANT:
#
# We DO NOT use:
#
#     cv2.VideoCapture()
#
# for the USB webcam.
#
# We directly read the MJPEG stream from V4L2.
# ============================================================

def usb_camera_capture():

    global usb_frame

    print()
    print("============================================")
    print("Initializing USB Webcam")
    print("============================================")

    print(
        f"Device     : {USB_CAMERA}"
    )

    print(
        f"Resolution : {USB_WIDTH}x{USB_HEIGHT}"
    )

    print(
        f"FPS        : {USB_FPS}"
    )

    print(
        "Format     : MJPEG"
    )

    print()


    # --------------------------------------------------------
    # Configure V4L2 camera
    # --------------------------------------------------------

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


    try:

        result = subprocess.run(

            configure_command,

            capture_output=True,

            text=True

        )


        if result.returncode != 0:

            print(
                "ERROR configuring USB camera:"
            )

            print(
                result.stderr
            )

            return


        print(
            "USB camera configured successfully"
        )

        print(
            result.stdout
        )


    except Exception as e:

        print(
            "Failed to configure USB camera:"
        )

        print(e)

        return


    # --------------------------------------------------------
    # Open V4L2 stream
    #
    # v4l2-ctl outputs raw MJPEG frames.
    # We read those frames directly.
    # --------------------------------------------------------

    command = [

        "v4l2-ctl",

        "-d",
        USB_CAMERA,

        "--stream-mmap",

        "--stream-poll"

    ]


    try:

        process = subprocess.Popen(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.DEVNULL,

            bufsize=0

        )


        print(
            "USB webcam streaming started"
        )


    except Exception as e:

        print(
            "Could not start USB webcam:"
        )

        print(e)

        return


    # --------------------------------------------------------
    # NOTE
    #
    # v4l2-ctl's --stream-mmap output is not a convenient
    # frame-delimited byte stream for a Flask application.
    #
    # Therefore we use a direct V4L2 reader below.
    # --------------------------------------------------------

    process.terminate()

    try:

        process.wait(
            timeout=1
        )

    except subprocess.TimeoutExpired:

        process.kill()


    # --------------------------------------------------------
    # Use ffmpeg as a V4L2 MJPEG reader.
    #
    # ffmpeg reads the webcam but does NOT decode/re-encode
    # the JPEG frames.
    #
    # It extracts the MJPEG packets and outputs them as
    # an image2pipe stream.
    # --------------------------------------------------------

    ffmpeg_command = [

        "ffmpeg",

        "-hide_banner",

        "-loglevel",
        "error",

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

        "-c:v",
        "copy",

        "-f",
        "mjpeg",

        "pipe:1"

    ]


    try:

        ffmpeg_process = subprocess.Popen(

            ffmpeg_command,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            bufsize=0

        )

    except Exception as e:

        print(
            "Could not start FFmpeg:"
        )

        print(e)

        return


    print(
        "USB MJPEG passthrough started"
    )


    # --------------------------------------------------------
    # Read JPEG frames
    # --------------------------------------------------------

    buffer = bytearray()

    frame_count = 0

    start_time = time.time()


    while True:

        try:

            data = ffmpeg_process.stdout.read(
                4096
            )


            if not data:

                print(
                    "USB webcam stream ended"
                )

                break


            buffer.extend(data)


            # ------------------------------------------------
            # Find JPEG Start Of Image
            # FFD8
            # ------------------------------------------------

            while True:

                start = buffer.find(
                    b"\xff\xd8"
                )


                if start == -1:

                    # Keep buffer from growing
                    if len(buffer) > 1_000_000:

                        buffer.clear()

                    break


                # Remove data before JPEG
                if start > 0:

                    del buffer[:start]


                # ------------------------------------------------
                # Find JPEG End Of Image
                # FFD9
                # ------------------------------------------------

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


                # ------------------------------------------------
                # Store latest JPEG
                # ------------------------------------------------

                with usb_lock:

                    usb_frame = jpeg


                # ------------------------------------------------
                # FPS measurement
                # ------------------------------------------------

                frame_count += 1

                elapsed = (
                    time.time() - start_time
                )


                if elapsed >= 5:

                    fps = (
                        frame_count /
                        elapsed
                    )

                    print(
                        "USB webcam FPS: "
                        f"{fps:.2f}"
                    )

                    frame_count = 0

                    start_time = time.time()


        except Exception as e:

            print(
                "USB webcam streaming error:"
            )

            print(e)

            break


# ============================================================
# IMX219 JPEG ENCODER
# ============================================================

def encode_imx_frame(frame):

    if frame is None:

        return None


    success, buffer = cv2.imencode(

        ".jpg",

        frame,

        [

            cv2.IMWRITE_JPEG_QUALITY,

            80

        ]

    )


    if not success:

        return None


    return buffer.tobytes()


# ============================================================
# IMX219 STREAM
# ============================================================

def imx_stream():

    while True:

        with imx_lock:

            frame = imx_frame

            if frame is not None:

                frame = frame.copy()


        if frame is None:

            time.sleep(0.05)

            continue


        jpeg = encode_imx_frame(
            frame
        )


        if jpeg is None:

            continue


        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n"

            b"Content-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"

            + jpeg

            + b"\r\n"

        )


        time.sleep(
            1 / IMX_STREAM_FPS
        )


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
# USB MJPEG STREAM
# ============================================================

def usb_stream():

    last_frame = None

    while True:

        with usb_lock:

            frame = usb_frame


        if frame is None:

            time.sleep(0.005)

            continue


        # Don't repeatedly send the same frame

        if frame == last_frame:

            time.sleep(0.001)

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
# LIDAR
# ============================================================

def lidar_reader():

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

        connection = mavutil.mavlink_connection(

            PIXHAWK_PORT,

            baud=PIXHAWK_BAUD

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


        # ----------------------------------------------------
        # Request DISTANCE_SENSOR at 10 Hz
        # ----------------------------------------------------

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
            "Requested DISTANCE_SENSOR "
            "at 10 Hz"
        )

        print(
            "----------------------------------------"
        )


        # ----------------------------------------------------
        # Read LiDAR
        # ----------------------------------------------------

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
# LIDAR API
# ============================================================

@app.route("/api/lidar")
def lidar_api():

    with lidar_lock:

        distance = lidar_distance_cm


    if distance is None:

        return jsonify({

            "distance_cm": None,

            "distance_m": None

        })


    return jsonify({

        "distance_cm": distance,

        "distance_m": (
            distance / 100.0
        )

    })


# ============================================================
# MAIN PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# START THREADS
# ============================================================

print()
print("============================================")
print("Starting threads")
print("============================================")


imx_thread = threading.Thread(

    target=imx219_capture,

    daemon=True

)


usb_thread = threading.Thread(

    target=usb_camera_capture,

    daemon=True

)


lidar_thread = threading.Thread(

    target=lidar_reader,

    daemon=True

)


imx_thread.start()

usb_thread.start()

lidar_thread.start()


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
