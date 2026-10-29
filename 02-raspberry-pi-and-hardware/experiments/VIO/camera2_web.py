import os
import time
import threading
import signal
from pathlib import Path
from datetime import datetime

import cv2
from flask import Flask, Response, render_template_string, send_from_directory, jsonify

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "drone_recordings"

MAX_RECORDING_SECONDS = 10 * 60

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000

# Camera 0
CAM0_MAIN_SIZE = (640, 480)
CAM0_PREVIEW_SIZE = (320, 240)
CAM0_FPS = 60

# Camera 1
CAM1_MAIN_SIZE = (1920, 1080)
CAM1_PREVIEW_SIZE = (640, 360)
CAM1_FPS = 30

# Recording bitrates
CAM0_BITRATE = 1_000_000
CAM1_BITRATE = 5_000_000

# JPEG quality for browser preview
JPEG_QUALITY = 70


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

BASE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CAMERA CLASS
# ============================================================

class CameraStream:

    def __init__(
        self,
        camera_num,
        main_size,
        preview_size,
        fps,
        bitrate,
        name
    ):

        self.camera_num = camera_num
        self.main_size = main_size
        self.preview_size = preview_size
        self.fps = fps
        self.bitrate = bitrate
        self.name = name

        self.picam2 = None

        self.running = False
        self.recording = False

        self.encoder = None
        self.output = None

        self.current_file = None

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.preview_thread = None

        self.recording_start_time = None

    # --------------------------------------------------------
    # CAMERA INITIALIZATION
    # --------------------------------------------------------

    def initialize(self):

        print(f"[{self.name}] Initializing camera...")

        self.picam2 = Picamera2(self.camera_num)

        config = self.picam2.create_video_configuration(

            main={
                "size": self.main_size,
                "format": "YUV420"
            },

            lores={
                "size": self.preview_size,
                "format": "RGB888"
            },

            controls={
                "FrameRate": self.fps
            }
        )

        self.picam2.configure(config)

        print(
            f"[{self.name}] Configuration:"
            f" main={self.main_size}@{self.fps}fps"
            f" preview={self.preview_size}"
        )

    # --------------------------------------------------------
    # START CAMERA
    # --------------------------------------------------------

    def start(self):

        if self.picam2 is None:
            self.initialize()

        print(f"[{self.name}] Starting camera...")

        self.picam2.start()

        time.sleep(1)

        self.running = True

        self.preview_thread = threading.Thread(
            target=self.preview_loop,
            daemon=True
        )

        self.preview_thread.start()

        print(f"[{self.name}] Camera started.")

    # --------------------------------------------------------
    # PREVIEW LOOP
    # --------------------------------------------------------

    def preview_loop(self):

        print(f"[{self.name}] Preview thread started.")

        while self.running:

            try:

                frame = self.picam2.capture_array("lores")

                success, jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        JPEG_QUALITY
                    ]
                )

                if success:

                    with self.frame_lock:

                        self.latest_frame = jpeg.tobytes()

            except Exception as e:

                print(
                    f"[{self.name}] Preview error: {e}"
                )

                time.sleep(0.1)

    # --------------------------------------------------------
    # GET LATEST JPEG
    # --------------------------------------------------------

    def get_frame(self):

        with self.frame_lock:

            return self.latest_frame

    # --------------------------------------------------------
    # START RECORDING
    # --------------------------------------------------------

    def start_recording(self, output_file):

        if self.recording:

            print(
                f"[{self.name}] Already recording."
            )

            return

        print(
            f"[{self.name}] Starting recording:"
            f" {output_file}"
        )

        self.current_file = output_file

        self.encoder = H264Encoder(
            bitrate=self.bitrate
        )

        self.output = FfmpegOutput(
            str(output_file)
        )

        self.picam2.start_recording(
            self.encoder,
            self.output
        )

        self.recording = True

        self.recording_start_time = time.time()

        print(
            f"[{self.name}] Recording started."
        )

    # --------------------------------------------------------
    # STOP RECORDING
    # --------------------------------------------------------

    def stop_recording(self):

        if not self.recording:

            return

        print(
            f"[{self.name}] Stopping recording..."
        )

        try:

            self.picam2.stop_recording()

        except Exception as e:

            print(
                f"[{self.name}] Stop recording error:"
                f" {e}"
            )

        self.recording = False

        self.recording_start_time = None

        self.encoder = None
        self.output = None

        print(
            f"[{self.name}] Recording stopped."
        )

    # --------------------------------------------------------
    # RECORDING DURATION
    # --------------------------------------------------------

    def recording_duration(self):

        if not self.recording:
            return 0

        if self.recording_start_time is None:
            return 0

        return int(
            time.time() - self.recording_start_time
        )

    # --------------------------------------------------------
    # STOP CAMERA
    # --------------------------------------------------------

    def stop(self):

        print(f"[{self.name}] Stopping camera...")

        self.stop_recording()

        self.running = False

        time.sleep(0.2)

        try:

            if self.picam2 is not None:
                self.picam2.stop()

        except Exception as e:

            print(
                f"[{self.name}] Camera stop error:"
                f" {e}"
            )

        try:

            if self.picam2 is not None:
                self.picam2.close()

        except Exception:
            pass

        print(f"[{self.name}] Camera stopped.")


# ============================================================
# CAMERA INSTANCES
# ============================================================

cam0 = CameraStream(
    camera_num=0,
    main_size=CAM0_MAIN_SIZE,
    preview_size=CAM0_PREVIEW_SIZE,
    fps=CAM0_FPS,
    bitrate=CAM0_BITRATE,
    name="CAMERA 0"
)

cam1 = CameraStream(
    camera_num=1,
    main_size=CAM1_MAIN_SIZE,
    preview_size=CAM1_PREVIEW_SIZE,
    fps=CAM1_FPS,
    bitrate=CAM1_BITRATE,
    name="CAMERA 1"
)


# ============================================================
# RECORDING STATE
# ============================================================

recording = False
flight_folder = None
recording_start_time = None

recording_lock = threading.Lock()


# ============================================================
# STORAGE
# ============================================================

def get_storage_info():

    try:

        stat = os.statvfs(BASE_DIR)

        total = stat.f_blocks * stat.f_frsize

        free = stat.f_bavail * stat.f_frsize

        used = total - free

        return {
            "total": total,
            "used": used,
            "free": free,
            "used_percent": (
                (used / total) * 100
                if total
                else 0
            )
        }

    except Exception:

        return {
            "total": 0,
            "used": 0,
            "free": 0,
            "used_percent": 0
        }


def format_size(size):

    if size < 1024:
        return f"{size} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"

    if size < 1024 ** 3:
        return f"{size / (1024 ** 2):.1f} MB"

    return f"{size / (1024 ** 3):.2f} GB"


# ============================================================
# RECORDING FUNCTIONS
# ============================================================

def start_recording():

    global recording
    global flight_folder
    global recording_start_time

    with recording_lock:

        if recording:

            return False, "Already recording."

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        flight_folder = (
            BASE_DIR /
            f"flight_{timestamp}"
        )

        flight_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        cam0_file = (
            flight_folder /
            "camera0_640x480_60fps.mp4"
        )

        cam1_file = (
            flight_folder /
            "camera1_1920x1080_30fps.mp4"
        )

        try:

            # Start Camera 0 recording
            cam0.start_recording(
                cam0_file
            )

            # Start Camera 1 recording
            cam1.start_recording(
                cam1_file
            )

            recording = True

            recording_start_time = time.time()

            print()
            print("=" * 60)
            print("RECORDING STARTED")
            print("=" * 60)
            print(f"Folder: {flight_folder}")
            print(f"Camera 0: {cam0_file}")
            print(f"Camera 1: {cam1_file}")
            print("=" * 60)
            print()

            return True, "Recording started."

        except Exception as e:

            print(
                f"Recording start error: {e}"
            )

            cam0.stop_recording()
            cam1.stop_recording()

            recording = False
            recording_start_time = None

            return False, str(e)


def stop_recording():

    global recording
    global recording_start_time

    with recording_lock:

        if not recording:

            return False, "Not recording."

        print()
        print("=" * 60)
        print("STOPPING RECORDING")
        print("=" * 60)

        try:

            # Stop both cameras
            cam0.stop_recording()
            cam1.stop_recording()

            recording = False
            recording_start_time = None

            print("Recording stopped.")
            print("=" * 60)
            print()

            return True, "Recording stopped."

        except Exception as e:

            print(
                f"Recording stop error: {e}"
            )

            recording = False
            recording_start_time = None

            return False, str(e)


# ============================================================
# AUTO STOP THREAD
# ============================================================

def recording_monitor():

    global recording

    while True:

        time.sleep(1)

        if not recording:
            continue

        if recording_start_time is None:
            continue

        elapsed = (
            time.time() -
            recording_start_time
        )

        if elapsed >= MAX_RECORDING_SECONDS:

            print()
            print(
                "Maximum recording time reached."
            )
            print(
                "Automatically stopping recording."
            )

            stop_recording()


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_frames(camera):

    while True:

        frame = camera.get_frame()

        if frame is None:

            time.sleep(0.05)

            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

        time.sleep(0.03)


# ============================================================
# WEB PAGE
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Drone Camera Dashboard</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #111;

    color: #fff;
}

.header {

    background: #1b1b1b;

    padding: 20px;

    text-align: center;

    border-bottom:
        1px solid #333;
}

.header h1 {

    margin: 0;

    font-size: 28px;
}

.header p {

    margin-top: 8px;

    color: #aaa;
}

.container {

    max-width: 1400px;

    margin: auto;

    padding: 20px;
}

.status-card {

    background: #1b1b1b;

    border-radius: 12px;

    padding: 20px;

    margin-bottom: 20px;

    display: flex;

    justify-content:
        space-between;

    align-items: center;

    flex-wrap: wrap;

    gap: 15px;
}

.status {

    font-size: 20px;

    font-weight: bold;
}

.status.recording {

    color: #ff4d4d;
}

.status.idle {

    color: #4caf50;
}

button {

    border: none;

    border-radius: 8px;

    padding: 14px 28px;

    font-size: 16px;

    font-weight: bold;

    cursor: pointer;
}

.start-btn {

    background: #2196f3;

    color: white;
}

.start-btn:hover {

    background: #1976d2;
}

.stop-btn {

    background: #f44336;

    color: white;
}

.stop-btn:hover {

    background: #d32f2f;
}

.camera-grid {

    display: grid;

    grid-template-columns:
        repeat(auto-fit, minmax(400px, 1fr));

    gap: 20px;

    margin-bottom: 20px;
}

.camera-card {

    background: #1b1b1b;

    border-radius: 12px;

    overflow: hidden;
}

.camera-title {

    padding: 15px;

    font-size: 18px;

    font-weight: bold;
}

.camera-feed {

    width: 100%;

    display: block;

    background: #000;
}

.camera-info {

    padding: 15px;

    color: #aaa;

    line-height: 1.6;
}

.info-grid {

    display: grid;

    grid-template-columns:
        repeat(auto-fit, minmax(250px, 1fr));

    gap: 20px;

    margin-bottom: 20px;
}

.info-card {

    background: #1b1b1b;

    padding: 20px;

    border-radius: 12px;
}

.info-card h3 {

    margin-top: 0;

    color: #aaa;
}

.info-value {

    font-size: 24px;

    font-weight: bold;
}

.recordings {

    background: #1b1b1b;

    padding: 20px;

    border-radius: 12px;
}

.recording {

    padding: 15px;

    border-bottom:
        1px solid #333;
}

.recording:last-child {

    border-bottom: none;
}

.recording-name {

    font-size: 18px;

    font-weight: bold;
}

.file {

    display: flex;

    justify-content:
        space-between;

    align-items: center;

    gap: 10px;

    padding: 8px 0;

    flex-wrap: wrap;
}

.download {

    background: #444;

    color: white;

    text-decoration: none;

    padding: 8px 14px;

    border-radius: 6px;
}

.download:hover {

    background: #555;
}

.message {

    margin-top: 15px;

    padding: 12px;

    border-radius: 8px;

    display: none;
}

@media(max-width: 600px) {

    .camera-grid {

        grid-template-columns: 1fr;
    }

    .container {

        padding: 10px;
    }

    .camera-grid {

        grid-template-columns:
            minmax(0, 1fr);
    }
}

</style>

</head>

<body>

<div class="header">

    <h1>🚁 Drone Camera Dashboard</h1>

    <p>Raspberry Pi 5 • Dual IMX219</p>

</div>


<div class="container">


    <div class="status-card">

        <div>

            <div
                id="status"
                class="status idle"
            >
                ● IDLE
            </div>

            <div id="duration">
                Duration: 00:00
            </div>

        </div>


        <div>

            <button
                id="startBtn"
                class="start-btn"
                onclick="startRecording()"
            >
                START RECORDING
            </button>


            <button
                id="stopBtn"
                class="stop-btn"
                onclick="stopRecording()"
                style="display:none"
            >
                STOP RECORDING
            </button>

        </div>

    </div>


    <div
        id="message"
        class="message"
    ></div>


    <div class="camera-grid">


        <div class="camera-card">

            <div class="camera-title">

                Camera 0

            </div>


            <img
                class="camera-feed"
                src="/video_feed/0"
            >


            <div class="camera-info">

                Recording:
                <strong>
                    640 × 480 @ 60 FPS
                </strong>

                <br>

                Live preview:
                <strong>
                    320 × 240
                </strong>

            </div>

        </div>


        <div class="camera-card">

            <div class="camera-title">

                Camera 1

            </div>


            <img
                class="camera-feed"
                src="/video_feed/1"
            >


            <div class="camera-info">

                Recording:
                <strong>
                    1920 × 1080 @ 30 FPS
                </strong>

                <br>

                Live preview:
                <strong>
                    640 × 360
                </strong>

            </div>

        </div>


    </div>


    <div class="info-grid">


        <div class="info-card">

            <h3>Storage Free</h3>

            <div
                id="storage"
                class="info-value"
            >
                --
            </div>

        </div>


        <div class="info-card">

            <h3>Recording Limit</h3>

            <div class="info-value">

                10:00

            </div>

        </div>


        <div class="info-card">

            <h3>Recording Folder</h3>

            <div
                class="info-value"
                style="font-size:16px"
            >
                ~/drone_recordings
            </div>

        </div>


    </div>


    <div class="recordings">

        <h2>Previous Recordings</h2>

        <div id="recordings">

            Loading...

        </div>

    </div>


</div>


<script>

let recordingStart = null;


function showMessage(message) {

    const box =
        document.getElementById("message");

    box.innerText = message;

    box.style.display = "block";

    setTimeout(() => {

        box.style.display = "none";

    }, 4000);

}


async function startRecording() {

    try {

        const response =
            await fetch("/api/start", {
                method: "POST"
            });

        const data =
            await response.json();

        showMessage(data.message);

        updateStatus();

    }

    catch(error) {

        showMessage(
            "Failed to start recording."
        );

    }

}


async function stopRecording() {

    try {

        const response =
            await fetch("/api/stop", {
                method: "POST"
            });

        const data =
            await response.json();

        showMessage(data.message);

        updateStatus();

        setTimeout(
            loadRecordings,
            1000
        );

    }

    catch(error) {

        showMessage(
            "Failed to stop recording."
        );

    }

}


async function updateStatus() {

    try {

        const response =
            await fetch("/api/status");

        const data =
            await response.json();


        const status =
            document.getElementById("status");

        const start =
            document.getElementById("startBtn");

        const stop =
            document.getElementById("stopBtn");


        if(data.recording) {

            status.innerText =
                "● RECORDING";

            status.className =
                "status recording";

            start.style.display =
                "none";

            stop.style.display =
                "inline-block";

            const seconds =
                data.duration;

            const minutes =
                Math.floor(seconds / 60);

            const secs =
                seconds % 60;

            document.getElementById(
                "duration"
            ).innerText =
                "Duration: " +
                String(minutes).padStart(2, "0") +
                ":" +
                String(secs).padStart(2, "0");

        }

        else {

            status.innerText =
                "● IDLE";

            status.className =
                "status idle";

            start.style.display =
                "inline-block";

            stop.style.display =
                "none";

            document.getElementById(
                "duration"
            ).innerText =
                "Duration: 00:00";

        }

    }

    catch(error) {

        console.log(error);

    }

}


async function updateStorage() {

    try {

        const response =
            await fetch("/api/storage");

        const data =
            await response.json();

        document.getElementById(
            "storage"
        ).innerText =
            data.free;

    }

    catch(error) {

        console.log(error);

    }

}


async function loadRecordings() {

    try {

        const response =
            await fetch("/api/recordings");

        const data =
            await response.json();


        const container =
            document.getElementById(
                "recordings"
            );


        if(data.length === 0) {

            container.innerHTML =
                "<p>No recordings yet.</p>";

            return;
        }


        container.innerHTML = "";


        data.forEach(folder => {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "recording";


            let html =
                `<div class="recording-name">
                    ${folder.name}
                </div>`;


            folder.files.forEach(file => {

                html += `

                    <div class="file">

                        <span>
                            ${file.name}
                            (${file.size})
                        </span>

                        <a
                            class="download"
                            href="${file.url}"
                        >
                            Download
                        </a>

                    </div>

                `;

            });


            div.innerHTML = html;

            container.appendChild(div);

        });

    }

    catch(error) {

        console.log(error);

    }

}


setInterval(
    updateStatus,
    1000
);

setInterval(
    updateStorage,
    5000
);

setInterval(
    loadRecordings,
    10000
);


updateStatus();
updateStorage();
loadRecordings();

</script>


</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML_PAGE
    )


@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):

    if camera_id == 0:

        camera = cam0

    elif camera_id == 1:

        camera = cam1

    else:

        return "Invalid camera", 404

    return Response(
        generate_frames(camera),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# API: START
# ============================================================

@app.route(
    "/api/start",
    methods=["POST"]
)
def api_start():

    success, message = start_recording()

    return jsonify({
        "success": success,
        "message": message
    })


# ============================================================
# API: STOP
# ============================================================

@app.route(
    "/api/stop",
    methods=["POST"]
)
def api_stop():

    success, message = stop_recording()

    return jsonify({
        "success": success,
        "message": message
    })


# ============================================================
# API: STATUS
# ============================================================

@app.route("/api/status")
def api_status():

    duration = 0

    if recording and recording_start_time:

        duration = int(
            time.time() -
            recording_start_time
        )

    return jsonify({

        "recording": recording,

        "duration": duration,

        "max_duration":
            MAX_RECORDING_SECONDS,

        "camera0_recording":
            cam0.recording,

        "camera1_recording":
            cam1.recording

    })


# ============================================================
# API: STORAGE
# ============================================================

@app.route("/api/storage")
def api_storage():

    info = get_storage_info()

    return jsonify({

        "total": format_size(
            info["total"]
        ),

        "used": format_size(
            info["used"]
        ),

        "free": format_size(
            info["free"]
        ),

        "used_percent":
            round(
                info["used_percent"],
                1
            )

    })


# ============================================================
# API: RECORDINGS
# ============================================================

@app.route("/api/recordings")
def api_recordings():

    recordings = []

    folders = sorted(
        BASE_DIR.glob("flight_*"),
        reverse=True
    )

    for folder in folders:

        if not folder.is_dir():
            continue

        files = []

        for file in sorted(
            folder.glob("*.mp4")
        ):

            try:

                size = file.stat().st_size

                relative_path = (
                    file.relative_to(BASE_DIR)
                )

                files.append({

                    "name": file.name,

                    "size":
                        format_size(size),

                    "url":
                        "/download/" +
                        str(relative_path)

                })

            except Exception:
                continue

        if files:

            recordings.append({

                "name": folder.name,

                "files": files

            })

    return jsonify(recordings)


# ============================================================
# DOWNLOAD
# ============================================================

@app.route(
    "/download/<path:filename>"
)
def download(filename):

    path = BASE_DIR / filename

    if not path.exists():

        return "File not found", 404

    if not path.is_file():

        return "Invalid file", 404

    return send_from_directory(
        BASE_DIR,
        filename,
        as_attachment=True
    )


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown():

    print()
    print("=" * 60)
    print("SHUTTING DOWN")
    print("=" * 60)

    try:

        stop_recording()

    except Exception:
        pass

    try:

        cam0.stop()

    except Exception:
        pass

    try:

        cam1.stop()

    except Exception:
        pass

    print("Shutdown complete.")


def signal_handler(
    sig,
    frame
):

    shutdown()

    os._exit(0)


signal.signal(
    signal.SIGINT,
    signal_handler
)

signal.signal(
    signal.SIGTERM,
    signal_handler
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DRONE CAMERA WEB DASHBOARD")
    print("=" * 60)

    print(
        f"Storage directory: {BASE_DIR}"
    )

    print(
        f"Camera 0: "
        f"{CAM0_MAIN_SIZE[0]}x{CAM0_MAIN_SIZE[1]}"
        f" @ {CAM0_FPS} FPS"
    )

    print(
        f"Camera 1: "
        f"{CAM1_MAIN_SIZE[0]}x{CAM1_MAIN_SIZE[1]}"
        f" @ {CAM1_FPS} FPS"
    )

    print(
        f"Web server: "
        f"http://<RASPBERRY_PI_IP>:{WEB_PORT}"
    )

    print("=" * 60)
    print()

    try:

        # Start both cameras.
        # They remain running so the live preview
        # is always available.

        cam0.start()

        cam1.start()

        print()
        print("Both cameras are ready.")
        print()

        # Recording watchdog
        monitor_thread = threading.Thread(
            target=recording_monitor,
            daemon=True
        )

        monitor_thread.start()

        # Start Flask
        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            threaded=True,
            debug=False,
            use_reloader=False
        )

    except KeyboardInterrupt:

        pass

    finally:

        shutdown()