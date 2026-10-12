#!/usr/bin/env python3
"""
CrackSense drone automation daemon.

No manual command is needed after the Raspberry Pi boots.
The daemon watches videos/incoming for completed video files,
runs the existing extraction+crack-detection script, then
moves the source video to processed/ or failed/.

Expected project layout:
    CrackSense/
    ├── drone_pipeline.py
    ├── extract_frames_with_crack_detection_2.py
    ├── Quantised/
    │   └── latest_int8.pt
    └── videos/
        ├── incoming/
        ├── processed/
        └── failed/
"""

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Your uploaded file already performs BOTH frame extraction
# and crack detection.
DETECTION_SCRIPT = (
    BASE_DIR / "extract_frames_with_crack_detection_2.py"
)

# IMPORTANT:
# Force INT8 on the Raspberry Pi. Do not rely on the script's
# automatic checkpoint search because that search checks FP32
# before FP16 and INT8.
CHECKPOINT = (
    BASE_DIR / "Quantised" / "latest_int8.pt"
)

INCOMING_DIR = (
    BASE_DIR / "videos" / "incoming"
)

PROCESSED_DIR = (
    BASE_DIR / "videos" / "processed"
)

FAILED_DIR = (
    BASE_DIR / "videos" / "failed"
)

RESULTS_DIR = (
    BASE_DIR / "results"
)

LOG_DIR = (
    BASE_DIR / "logs"
)

# Process 2 frames/second, matching your existing script's default.
PROCESS_FPS = 2.0

# A recording file must stop changing for this long before
# we process it.
STABLE_SECONDS = 5

# Polling interval while waiting for a new video.
POLL_SECONDS = 1

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
}


# ============================================================
# DIRECTORIES
# ============================================================

for directory in (
    INCOMING_DIR,
    PROCESSED_DIR,
    FAILED_DIR,
    RESULTS_DIR,
    LOG_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


LOG_FILE = (
    LOG_DIR / "drone_pipeline.log"
)


# ============================================================
# LOGGING
# ============================================================

def log(message):
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = (
        f"[{timestamp}] {message}"
    )

    print(
        line,
        flush=True,
    )

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            line + "\n"
        )


# ============================================================
# FILE HELPERS
# ============================================================

def video_files():
    files = []

    for path in INCOMING_DIR.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        # Ignore temporary/incomplete files if a recorder happens
        # to use these suffixes.
        if path.name.endswith(".part"):
            continue

        if path.name.endswith(".tmp"):
            continue

        files.append(path)

    return sorted(
        files,
        key=lambda p: p.stat().st_mtime,
    )


def wait_until_stable(path):
    """
    Wait until the video file size remains unchanged for
    STABLE_SECONDS.

    This prevents the detector from opening a video while
    the camera is still writing it.
    """

    log(
        f"Waiting for recording to finish: {path.name}"
    )

    previous_size = -1
    stable_since = None

    while True:

        if not path.exists():
            log(
                f"Video disappeared while waiting: {path.name}"
            )
            return False

        try:
            current_size = path.stat().st_size
        except OSError:
            time.sleep(POLL_SECONDS)
            continue

        if current_size > 0 and current_size == previous_size:

            if stable_since is None:
                stable_since = time.monotonic()

            if (
                time.monotonic() - stable_since
                >= STABLE_SECONDS
            ):
                log(
                    f"Recording finished: {path.name} "
                    f"({current_size / 1024 / 1024:.1f} MB)"
                )
                return True

        else:
            stable_since = None

        previous_size = current_size

        time.sleep(POLL_SECONDS)


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):
    name = video_path.stem

    # Each video gets its own result root.
    video_result_dir = (
        RESULTS_DIR / name
    )

    video_result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("")
    log("=" * 70)
    log(f"NEW VIDEO: {video_path.name}")
    log("=" * 70)

    if not wait_until_stable(video_path):
        return

    if not CHECKPOINT.exists():
        log(
            f"ERROR: INT8 checkpoint not found: {CHECKPOINT}"
        )
        return

    if not DETECTION_SCRIPT.exists():
        log(
            f"ERROR: Detection script not found: "
            f"{DETECTION_SCRIPT}"
        )
        return

    command = [
        sys.executable,
        str(DETECTION_SCRIPT),

        "--video",
        str(video_path),

        "--output",
        str(video_result_dir),

        "--checkpoint",
        str(CHECKPOINT),

        "--fps",
        str(PROCESS_FPS),
    ]

    log(
        "Starting CrackSense extraction + detection..."
    )

    log(
        "Checkpoint: "
        + str(CHECKPOINT)
    )

    log(
        "Processing FPS: "
        + str(PROCESS_FPS)
    )

    log(
        "Command: "
        + " ".join(
            f'"{x}"' if " " in x else x
            for x in command
        )
    )

    start = time.monotonic()

    try:
        result = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            check=False,
        )

    except Exception as exc:
        log(
            f"ERROR launching detector: {exc}"
        )
        return

    elapsed = (
        time.monotonic() - start
    )

    if result.returncode == 0:

        destination = (
            PROCESSED_DIR /
            video_path.name
        )

        # Avoid overwriting an existing file.
        if destination.exists():
            stamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            destination = (
                PROCESSED_DIR /
                f"{video_path.stem}_{stamp}"
                f"{video_path.suffix}"
            )

        try:
            shutil.move(
                str(video_path),
                str(destination),
            )

            log(
                f"Video moved to: {destination}"
            )

        except Exception as exc:
            log(
                f"WARNING: Could not move processed "
                f"video: {exc}"
            )

        log(
            f"CrackSense completed successfully "
            f"in {elapsed:.1f} seconds."
        )

        log(
            f"Results: {video_result_dir}"
        )

    else:

        destination = (
            FAILED_DIR /
            video_path.name
        )

        if destination.exists():
            stamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            destination = (
                FAILED_DIR /
                f"{video_path.stem}_{stamp}"
                f"{video_path.suffix}"
            )

        try:
            shutil.move(
                str(video_path),
                str(destination),
            )

        except Exception as exc:
            log(
                f"WARNING: Could not move failed "
                f"video: {exc}"
            )

        log(
            f"ERROR: CrackSense returned exit code "
            f"{result.returncode}."
        )

        log(
            f"Failed video moved to: {destination}"
        )

    log("=" * 70)
    log("")


# ============================================================
# MAIN DAEMON LOOP
# ============================================================

def main():

    log("")
    log("=" * 70)
    log("CRACKSENSE DRONE PIPELINE STARTED")
    log("=" * 70)
    log(
        f"Project directory: {BASE_DIR}"
    )
    log(
        f"Watching: {INCOMING_DIR}"
    )
    log(
        f"INT8 checkpoint: {CHECKPOINT}"
    )
    log(
        f"Detection script: {DETECTION_SCRIPT}"
    )
    log(
        "Waiting for videos..."
    )

    while True:

        try:
            files = video_files()

            if files:
                # Process one at a time.
                process_video(files[0])

            else:
                time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            log(
                "Shutdown requested."
            )
            break

        except Exception as exc:
            log(
                f"Pipeline loop error: {exc}"
            )

            # Keep the drone service alive even if one unexpected
            # filesystem error occurs.
            time.sleep(5)


if __name__ == "__main__":
    main()
