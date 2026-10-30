#!/usr/bin/env python3
"""Minimal ArUco -> MAVLink ODOMETRY bench test for ArduPilot EKF3.

Safety properties:
* Does not arm, change modes, or command movement.
* Does not transmit unless --send is supplied.
* Stops publishing when the selected marker is lost or pose checks fail.

Coordinate conventions:
* World: Local FRD/NED-like frame: X forward/north, Y right/east, Z down.
* Body:  FRD: X forward, Y right, Z down.
* Camera: OpenCV: X image-right, Y image-down, Z optical-forward.
* Marker: X printed-right, Y printed-up, Z out of the printed face.

Initial physical assumptions:
* Marker lies flat on the floor at world (0, 0, 0).
* Printed top of marker points toward world +X.
* Camera points vertically downward.
* Top of camera image points toward vehicle +X (front).
* Camera lens is at the vehicle origin unless offsets are supplied.

Install on Raspberry Pi OS:
    sudo apt install python3-opencv python3-picamera2
    python3 -m pip install pymavlink

Dry run first:
    python3 aruco_externalnav_test.py --marker-id 0

Transmit after validating every axis with propellers removed:
    python3 aruco_externalnav_test.py --marker-id 0 --device /dev/serial0 \
        --baud 115200 --send
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from picamera2 import Picamera2

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None


# Calibration values copied from the user's tested 640x480 detector.
WIDTH = 640
HEIGHT = 480
CAMERA_MATRIX = np.array(
    [[890.241265, 0.0, 377.331918],
     [0.0, 887.723614, 200.943233],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DIST_COEFFS = np.array(
    [-0.36967095, 0.3668438, 0.00159773, 0.00323534, -0.6660792],
    dtype=np.float64,
)


@dataclass
class Pose:
    timestamp: float
    position: np.ndarray       # world position of body, metres, Local FRD
    rotation: np.ndarray       # R_W_B
    reprojection_error: float  # pixels RMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--marker-size", type=float, default=0.059,
                        help="printed black-square side length in metres")
    parser.add_argument("--device", default="/dev/serial0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--send", action="store_true",
                        help="actually transmit ODOMETRY; otherwise dry-run")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="maximum ODOMETRY publication rate in Hz")
    parser.add_argument("--reprojection-limit", type=float, default=3.0,
                        help="reject pose above this RMS pixel error")
    parser.add_argument("--max-jump-speed", type=float, default=3.0,
                        help="reject implied translation faster than m/s")
    parser.add_argument("--filter-alpha", type=float, default=0.35,
                        help="position EMA weight for new sample, 0 < a <= 1")
    parser.add_argument("--camera-forward", type=float, default=0.0,
                        help="camera lens forward of body origin, metres")
    parser.add_argument("--camera-right", type=float, default=0.0,
                        help="camera lens right of body origin, metres")
    parser.add_argument("--camera-down", type=float, default=0.0,
                        help="camera lens below body origin, metres")
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0,
                        help="clockwise camera mounting correction; 0 means image top is vehicle front")
    parser.add_argument("--marker-heading-deg", type=float, default=0.0,
                        help="clockwise heading of printed marker top from world +X")
    return parser.parse_args()


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return result


def rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def matrix_to_quaternion_wxyz(r: np.ndarray) -> list[float]:
    """Convert a proper 3x3 rotation matrix to normalized [w,x,y,z]."""
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s,
                      (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                          (r[0, 1] + r[1, 0]) / s,
                          (r[0, 2] + r[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 2] - r[2, 0]) / s,
                          (r[0, 1] + r[1, 0]) / s, 0.25 * s,
                          (r[1, 2] + r[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[1, 0] - r[0, 1]) / s,
                          (r[0, 2] + r[2, 0]) / s,
                          (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q.tolist()


def yaw_from_rotation(r: np.ndarray) -> float:
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def marker_world_transform(marker_heading_deg: float) -> np.ndarray:
    # At zero heading: marker +Y (printed up) is world +X, marker +X
    # (printed right) is world +Y, and marker +Z is world -Z (up).
    r_w_m_zero = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    r_w_m = rotation_z(math.radians(marker_heading_deg)) @ r_w_m_zero
    return transform(r_w_m, np.zeros(3))


def body_camera_transform(args: argparse.Namespace) -> np.ndarray:
    # At zero correction and with image top facing vehicle front:
    # camera +X -> body +Y, camera +Y -> body -X, camera +Z -> body +Z.
    r_b_c_zero = np.array(
        [[0.0, -1.0, 0.0],
         [1.0,  0.0, 0.0],
         [0.0,  0.0, 1.0]],
        dtype=np.float64,
    )
    r_b_c = rotation_z(math.radians(args.camera_yaw_deg)) @ r_b_c_zero
    t_b_c = np.array([args.camera_forward, args.camera_right, args.camera_down])
    return transform(r_b_c, t_b_c)


def marker_object_points(marker_size: float) -> np.ndarray:
    half = marker_size / 2.0
    # Required IPPE_SQUARE order: top-left, top-right, bottom-right, bottom-left.
    return np.array(
        [[-half, half, 0.0],
         [half, half, 0.0],
         [half, -half, 0.0],
         [-half, -half, 0.0]],
        dtype=np.float64,
    )


def estimate_pose(
    image_points: np.ndarray,
    object_points: np.ndarray,
    t_w_m: np.ndarray,
    t_b_c: np.ndarray,
) -> Pose | None:
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        CAMERA_MATRIX,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS
    )
    residual = image_points.reshape(-1, 2) - projected.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

    r_c_m, _ = cv2.Rodrigues(rvec)
    t_c_m = transform(r_c_m, tvec.reshape(3))
    t_w_c = t_w_m @ np.linalg.inv(t_c_m)
    t_w_b = t_w_c @ np.linalg.inv(t_b_c)

    return Pose(
        timestamp=time.monotonic(),
        position=t_w_b[:3, 3].copy(),
        rotation=t_w_b[:3, :3].copy(),
        reprojection_error=rms,
    )


def make_covariance(position_sigma: float = 0.10,
                    angle_sigma_rad: float = 0.20) -> list[float]:
    # MAVLink upper-right triangle representation (21 elements).
    covariance = [0.0] * 21
    covariance[0] = position_sigma
    covariance[6] = position_sigma
    covariance[11] = position_sigma * 1.5
    covariance[15] = angle_sigma_rad
    covariance[18] = angle_sigma_rad
    covariance[20] = angle_sigma_rad
    return covariance


def connect_mavlink(args: argparse.Namespace):
    if not args.send:
        return None
    if mavutil is None:
        raise RuntimeError("pymavlink is missing: python3 -m pip install pymavlink")
    print(f"Connecting to {args.device} at {args.baud} baud...")
    link = mavutil.mavlink_connection(args.device, baud=args.baud)
    link.wait_heartbeat(timeout=15)
    print(f"Heartbeat: system={link.target_system}, component={link.target_component}")
    return link


def send_odometry(link, pose: Pose, quality: int, reset_counter: int = 0) -> None:
    nan = float("nan")
    q = matrix_to_quaternion_wxyz(pose.rotation)
    pose_covariance = make_covariance()
    velocity_covariance = [nan] * 21

    link.mav.odometry_send(
        int(time.monotonic_ns() // 1000),
        mavutil.mavlink.MAV_FRAME_LOCAL_FRD,
        mavutil.mavlink.MAV_FRAME_BODY_FRD,
        float(pose.position[0]),
        float(pose.position[1]),
        float(pose.position[2]),
        q,
        nan, nan, nan,              # velocity unused in initial EKF config
        nan, nan, nan,              # angular rates unused
        pose_covariance,
        velocity_covariance,
        reset_counter,
        mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION,
        int(quality),
    )


def main() -> int:
    args = parse_args()
    if not (0.0 < args.filter_alpha <= 1.0):
        raise ValueError("--filter-alpha must be in (0, 1]")
    if args.marker_size <= 0.0 or args.rate < 4.0:
        raise ValueError("marker size must be positive and rate must be >= 4 Hz")

    link = connect_mavlink(args)
    mode = "TRANSMIT" if args.send else "DRY RUN"
    print(f"Mode: {mode}; marker ID: {args.marker_id}")
    print("Assumption: image top points toward vehicle front.")
    print("Press Ctrl+C to stop. Never test with propellers installed.")

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )
    obj_points = marker_object_points(args.marker_size)
    t_w_m = marker_world_transform(args.marker_heading_deg)
    t_b_c = body_camera_transform(args)

    camera = Picamera2()
    camera.configure(camera.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
        controls={"FrameRate": 30},
    ))
    camera.start()
    time.sleep(2.0)

    running = True

    def stop_handler(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    filtered_position = None
    previous_raw_pose = None
    previous_valid_time = 0.0
    last_send_time = 0.0
    last_print_time = 0.0
    loss_reported = False
    interval = 1.0 / args.rate

    try:
        while running:
            frame = camera.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            now = time.monotonic()
            candidate = None

            if ids is not None:
                flat_ids = ids.flatten().tolist()
                if args.marker_id in flat_ids:
                    index = flat_ids.index(args.marker_id)
                    candidate = estimate_pose(
                        np.asarray(corners[index][0], dtype=np.float64),
                        obj_points,
                        t_w_m,
                        t_b_c,
                    )

            rejection = None
            if candidate is None:
                rejection = "marker missing"
            elif candidate.reprojection_error > args.reprojection_limit:
                rejection = f"reprojection {candidate.reprojection_error:.2f}px"
            elif previous_raw_pose is not None:
                dt = candidate.timestamp - previous_raw_pose.timestamp
                speed = np.linalg.norm(candidate.position - previous_raw_pose.position) / max(dt, 1e-3)
                if speed > args.max_jump_speed:
                    rejection = f"jump {speed:.2f}m/s"

            if rejection is not None:
                if not loss_reported and now - previous_valid_time > 0.25:
                    print(f"NO VALID POSE: {rejection}; publishing stopped")
                    loss_reported = True
                continue

            loss_reported = False
            previous_valid_time = now
            previous_raw_pose = candidate

            if filtered_position is None:
                filtered_position = candidate.position.copy()
            else:
                a = args.filter_alpha
                filtered_position = a * candidate.position + (1.0 - a) * filtered_position

            output_pose = Pose(
                timestamp=candidate.timestamp,
                position=filtered_position.copy(),
                rotation=candidate.rotation,
                reprojection_error=candidate.reprojection_error,
            )

            if now - last_send_time >= interval:
                # Simple quality score derived from reprojection error.
                quality = int(np.clip(100.0 - 20.0 * candidate.reprojection_error, 1, 100))
                if link is not None:
                    send_odometry(link, output_pose, quality)
                last_send_time = now

            if now - last_print_time >= 0.2:
                x, y, z = output_pose.position
                yaw = yaw_from_rotation(output_pose.rotation)
                prefix = "SENT" if link is not None else "POSE"
                print(
                    f"{prefix} X={x:+.3f} Y={y:+.3f} Z={z:+.3f}m "
                    f"yaw={yaw:+.1f}deg err={candidate.reprojection_error:.2f}px"
                )
                last_print_time = now
    finally:
        camera.stop()
        if link is not None:
            link.close()
        print("Stopped cleanly.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
