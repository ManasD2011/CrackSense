#!/usr/bin/env python3

import os
import signal
import subprocess
import threading
import time

from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template_string,
    send_from_directory,
    redirect,
    url_for,
    abort
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "drone_recordings"

HOST = "0.0.0.0"
PORT = 5000

MAX_RECORDING_SECONDS = 10 * 60


CAMERA_0 = {
    "id": 0,
    "width": 640,
    "height": 480,
    "fps": 60,
    "filename": "camera0_640x480_60fps.mp4"
}


CAMERA_1 = {
    "id": 1,
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "filename": "camera1_1920x1080_30fps.mp4"
}


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# RECORDING STATE
# ============================================================

processes = []

recording = False

start_time = None

flight_dir = None

state_lock = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def format_size(size_bytes):

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB"
    ]

    size = float(size_bytes)

    for unit in units:

        if size < 1024:

            return f"{size:.1f} {unit}"

        size /= 1024

    return f"{size:.1f} PB"


def get_storage():

    stat = os.statvfs(BASE_DIR)

    total = stat.f_blocks * stat.f_frsize

    free = stat.f_bavail * stat.f_frsize

    used = total - free

    percent = (
        used / total * 100
        if total
        else 0
    )

    return {
        "total": total,
        "used": used,
        "free": free,
        "percent": percent,
        "total_text": format_size(total),
        "used_text": format_size(used),
        "free_text": format_size(free)
    }


def get_recording_duration():

    if not recording or start_time is None:

        return 0

    return time.time() - start_time


def get_recordings():

    BASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    flights = []

    for directory in BASE_DIR.iterdir():

        if not directory.is_dir():

            continue

        if not directory.name.startswith("flight_"):

            continue

        files = []

        total_size = 0

        for file in sorted(directory.iterdir()):

            if file.is_file():

                size = file.stat().st_size

                total_size += size

                files.append({
                    "name": file.name,
                    "size": format_size(size)
                })

        flights.append({
            "name": directory.name,
            "files": files,
            "size": format_size(total_size)
        })

    flights.sort(
        key=lambda x: x["name"],
        reverse=True
    )

    return flights


# ============================================================
# CAMERA PROCESS
# ============================================================

def start_camera(camera):

    output_file = flight_dir / camera["filename"]

    command = [
        "rpicam-vid",

        "-t", "0",

        "--camera",
        str(camera["id"]),

        "--width",
        str(camera["width"]),

        "--height",
        str(camera["height"]),

        "--framerate",
        str(camera["fps"]),

        "--codec",
        "libav",

        "--libav-format",
        "mp4",

        "--output",
        str(output_file),

        "--nopreview"
    ]

    print(
        "Starting:",
        " ".join(command)
    )

    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )


# ============================================================
# STOP CAMERA
# ============================================================

def stop_camera(process):

    if process is None:

        return

    if process.poll() is None:

        try:

            process.send_signal(
                signal.SIGINT
            )

            process.wait(
                timeout=10
            )

        except subprocess.TimeoutExpired:

            print(
                "Camera did not stop "
                "cleanly. Terminating."
            )

            process.terminate()

            try:

                process.wait(
                    timeout=3
                )

            except subprocess.TimeoutExpired:

                process.kill()


# ============================================================
# START RECORDING
# ============================================================

def start_recording():

    global processes
    global recording
    global start_time
    global flight_dir

    with state_lock:

        if recording:

            return False, "Recording already running."

        BASE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        flight_dir = (
            BASE_DIR /
            f"flight_{timestamp}"
        )

        flight_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print(
            f"Starting recording: "
            f"{flight_dir}"
        )

        try:

            # Start camera 0
            cam0 = start_camera(
                CAMERA_0
            )

            # Small delay before camera 1
            time.sleep(0.5)

            # Start camera 1
            cam1 = start_camera(
                CAMERA_1
            )

            processes = [
                cam0,
                cam1
            ]

            start_time = time.time()

            recording = True

            print(
                "Dual camera recording started."
            )

            return True, "Recording started."

        except Exception as e:

            print(
                "ERROR starting cameras:",
                e
            )

            for process in processes:

                stop_camera(process)

            processes = []

            recording = False

            return False, str(e)


# ============================================================
# STOP RECORDING
# ============================================================

def stop_recording():

    global processes
    global recording
    global start_time

    with state_lock:

        if not recording:

            return False, "No recording running."

        print(
            "Stopping recording..."
        )

        for process in processes:

            stop_camera(process)

        processes = []

        duration = (
            time.time() - start_time
            if start_time
            else 0
        )

        recording = False

        start_time = None

        print(
            f"Recording stopped "
            f"after {duration:.1f} seconds."
        )

        return True, "Recording stopped."


# ============================================================
# AUTO STOP
# ============================================================

def recording_monitor():

    while True:

        time.sleep(1)

        if recording:

            duration = get_recording_duration()

            if duration >= MAX_RECORDING_SECONDS:

                print(
                    "Maximum recording "
                    "duration reached."
                )

                stop_recording()


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
    content="width=device-width, initial-scale=1.0"
>

<title>Drone Camera Recorder</title>

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

    background: #111827;

    color: #f9fafb;
}

.container {

    max-width: 1100px;

    margin: auto;

    padding: 25px;
}

.header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 25px;
}

h1 {

    margin: 0;

    font-size: 28px;
}

.card {

    background: #1f2937;

    border-radius: 12px;

    padding: 20px;

    margin-bottom: 20px;
}

.cameras {

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(280px, 1fr)
        );

    gap: 20px;
}

.camera {

    background: #111827;

    border-radius: 10px;

    padding: 20px;
}

.camera h2 {

    margin-top: 0;

    font-size: 20px;
}

.spec {

    color: #9ca3af;

    margin-bottom: 10px;
}

.status {

    display: inline-block;

    padding: 7px 12px;

    border-radius: 20px;

    font-weight: bold;

    font-size: 14px;
}

.ready {

    background: #065f46;

    color: #d1fae5;
}

.recording {

    background: #991b1b;

    color: #fee2e2;
}

button {

    border: none;

    border-radius: 8px;

    padding: 13px 22px;

    font-size: 16px;

    cursor: pointer;

    margin-right: 10px;

    margin-top: 10px;
}

.start {

    background: #16a34a;

    color: white;
}

.stop {

    background: #dc2626;

    color: white;
}

.delete {

    background: #7f1d1d;

    color: white;
}

button:disabled {

    opacity: 0.4;

    cursor: not-allowed;
}

.storage-bar {

    width: 100%;

    height: 20px;

    background: #374151;

    border-radius: 10px;

    overflow: hidden;

    margin-top: 10px;
}

.storage-used {

    height: 100%;

    background: #2563eb;

    transition: width 0.5s;
}

.flight {

    border-top: 1px solid #374151;

    padding: 15px 0;
}

.file {

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 8px 0;
}

.download {

    background: #2563eb;

    color: white;

    text-decoration: none;

    padding: 7px 12px;

    border-radius: 6px;
}

.small {

    color: #9ca3af;

    font-size: 14px;
}

</style>

</head>


<body>

<div class="container">


<div class="header">

<h1>Drone Camera Recorder</h1>

<span
    id="status"
    class="status ready"
>
READY
</span>

</div>


<!-- CAMERAS -->

<div class="cameras">


<div class="camera">

<h2>Camera 0</h2>

<div class="spec">

640 × 480 @ 60 FPS

</div>

<div>

H.264 / MP4

</div>

</div>


<div class="camera">

<h2>Camera 1</h2>

<div class="spec">

1920 × 1080 @ 30 FPS

</div>

<div>

H.264 / MP4

</div>

</div>


</div>


<!-- CONTROLS -->

<div class="card">

<h2>Recording Control</h2>

<div>

Duration:

<strong id="duration">
00:00
</strong>

</div>

<button
    id="startButton"
    class="start"
    onclick="startRecording()"
>
START RECORDING
</button>

<button
    id="stopButton"
    class="stop"
    onclick="stopRecording()"
    disabled
>
STOP RECORDING
</button>

</div>


<!-- STORAGE -->

<div class="card">

<h2>Storage</h2>

<div id="storageText">
Loading...
</div>

<div class="storage-bar">

<div
    id="storageBar"
    class="storage-used"
    style="width: 0%"
>
</div>

</div>

</div>


<!-- RECORDINGS -->

<div class="card">

<h2>Recordings</h2>

<div id="recordings">

Loading...

</div>

</div>


</div>


<script>


let recordingStart = null;


async function getStatus() {

    try {

        const response =
            await fetch("/api/status");

        const data =
            await response.json();


        const status =
            document.getElementById(
                "status"
            );


        const startButton =
            document.getElementById(
                "startButton"
            );


        const stopButton =
            document.getElementById(
                "stopButton"
            );


        if (data.recording) {

            status.textContent =
                "● RECORDING";

            status.className =
                "status recording";

            startButton.disabled =
                true;

            stopButton.disabled =
                false;

            recordingStart =
                data.start_time;

        } else {

            status.textContent =
                "READY";

            status.className =
                "status ready";

            startButton.disabled =
                false;

            stopButton.disabled =
                true;

            recordingStart = null;
        }


        updateDuration(
            data.duration
        );


        updateStorage(
            data.storage
        );


    } catch (error) {

        console.error(error);

    }

}


function updateDuration(seconds) {

    seconds =
        Math.floor(seconds || 0);


    const minutes =
        Math.floor(
            seconds / 60
        );


    const secs =
        seconds % 60;


    document.getElementById(
        "duration"
    ).textContent =

        String(minutes)
            .padStart(2, "0")

        + ":" +

        String(secs)
            .padStart(2, "0");
}


function updateStorage(storage) {

    document.getElementById(
        "storageText"
    ).textContent =

        storage.free_text
        + " free / "
        + storage.total_text
        + " total";


    document.getElementById(
        "storageBar"
    ).style.width =

        storage.percent + "%";
}


async function startRecording() {

    const response =
        await fetch(
            "/api/start",
            {
                method: "POST"
            }
        );


    const data =
        await response.json();


    if (!data.success) {

        alert(data.message);

    }


    refresh();

}


async function stopRecording() {

    const response =
        await fetch(
            "/api/stop",
            {
                method: "POST"
            }
        );


    const data =
        await response.json();


    if (!data.success) {

        alert(data.message);

    }


    refresh();

}


async function loadRecordings() {

    const response =
        await fetch(
            "/api/recordings"
        );


    const flights =
        await response.json();


    const container =
        document.getElementById(
            "recordings"
        );


    if (flights.length === 0) {

        container.innerHTML =
            '<div class="small">No recordings yet.</div>';

        return;
    }


    let html = "";


    for (const flight of flights) {

        html +=
            '<div class="flight">';

        html +=
            '<h3>' +
            flight.name +
            '</h3>';

        html +=
            '<div class="small">' +
            'Total size: ' +
            flight.size +
            '</div>';


        for (const file of flight.files) {

            html +=
                '<div class="file">';

            html +=
                '<span>' +
                file.name +
                ' (' +
                file.size +
                ')' +
                '</span>';


            html +=
                '<a class="download" ' +
                'href="/download/' +
                encodeURIComponent(
                    flight.name
                ) +
                '/' +
                encodeURIComponent(
                    file.name
                ) +
                '">' +
                'Download' +
                '</a>';


            html +=
                '</div>';
        }


        html +=
            '</div>';
    }


    container.innerHTML =
        html;
}


async function refresh() {

    await getStatus();

    await loadRecordings();

}


setInterval(
    refresh,
    1000
);


refresh();


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
        HTML
    )


# ------------------------------------------------------------
# STATUS
# ------------------------------------------------------------

@app.route("/api/status")
def status():

    storage = get_storage()

    return jsonify({

        "recording": recording,

        "duration":
            get_recording_duration(),

        "start_time":
            start_time,

        "storage":
            storage

    })


# ------------------------------------------------------------
# START
# ------------------------------------------------------------

@app.route(
    "/api/start",
    methods=["POST"]
)
def api_start():

    success, message = (
        start_recording()
    )

    return jsonify({

        "success": success,

        "message": message

    })


# ------------------------------------------------------------
# STOP
# ------------------------------------------------------------

@app.route(
    "/api/stop",
    methods=["POST"]
)
def api_stop():

    success, message = (
        stop_recording()
    )

    return jsonify({

        "success": success,

        "message": message

    })


# ------------------------------------------------------------
# RECORDINGS
# ------------------------------------------------------------

@app.route("/api/recordings")
def api_recordings():

    return jsonify(
        get_recordings()
    )


# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------

@app.route(
    "/download/<flight>/<filename>"
)
def download_file(
    flight,
    filename
):

    directory = (
        BASE_DIR /
        flight
    )

    file_path = (
        directory /
        filename
    )


    # Prevent path traversal
    try:

        file_path.resolve().relative_to(
            BASE_DIR.resolve()
        )

    except ValueError:

        abort(403)


    if not file_path.is_file():

        abort(404)


    return send_from_directory(
        directory,
        filename,
        as_attachment=True
    )


# ============================================================
# START MONITOR THREAD
# ============================================================

monitor_thread = threading.Thread(
    target=recording_monitor,
    daemon=True
)

monitor_thread.start()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    BASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("DRONE CAMERA WEB SERVER")
    print("=" * 60)
    print()
    print(
        f"Open from your computer:"
    )
    print()
    print(
        f"http://<RASPBERRY_PI_IP>:{PORT}"
    )
    print()
    print(
        "Press CTRL+C to stop the server."
    )
    print("=" * 60)
    print()

    app.run(
        host=HOST,
        port=PORT,
        threaded=True
    )
