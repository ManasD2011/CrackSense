#!/usr/bin/env python3
"""Read Pixhawk lidar telemetry. No arming, mode or movement commands.

Run in the same Python environment as heartbeat_test.py:
    python3 lidar_mavlink_test.py
    python3 lidar_mavlink_test.py --csv lidar_readings.csv

Stop other programs reading the same physical serial port first.
Orientations describe configured mounting, which must be verified physically.
Pixhawk message time and Pi reception time are NOT camera exposure times.
"""

import argparse
import csv
import time
from collections import deque


def label(orientation):
    return {0: "FORWARD / WALL", 25: "DOWN / GROUND"}.get(
        orientation, f"OTHER orientation={orientation}"
    )


def reading_status(msg):
    distance = int(msg.current_distance)
    lower, upper = int(msg.min_distance), int(msg.max_distance)
    if distance <= 0 or distance >= 65532:
        return "INVALID"
    if upper <= lower or lower < 0:
        return "LIMITS UNKNOWN"
    if not lower <= distance <= upper:
        return "OUT OF RANGE"
    if getattr(msg, "signal_quality", 0) == 1:
        return "INVALID QUALITY"
    return "IN RANGE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--rate", type=int, default=10,
                        help="Requested DISTANCE_SENSOR rate; actual rate is measured")
    parser.add_argument("--stale", type=float, default=1.0,
                        help="Pi reception age in seconds before STALE")
    parser.add_argument("--csv", help="Optional path; creates a NEW CSV, never overwrites")
    args = parser.parse_args()
    if not 1 <= args.rate <= 50 or not 0 < args.stale <= 60 or args.baud <= 0:
        parser.error("Use rate 1..50, stale timeout 0..60 seconds, and positive baud")

    from pymavlink import mavutil

    master = None
    log = None
    try:
        if args.csv:
            log = open(args.csv, "x", newline="", encoding="utf-8")
        writer = csv.writer(log) if log else None
        if writer:
            writer.writerow([
                "pi_receive_monotonic_ns", "pixhawk_time_boot_ms",
                "system_id", "component_id", "sensor_id", "orientation",
                "raw_distance_cm", "distance_m", "min_cm", "max_cm", "status"
            ])

        master = mavutil.mavlink_connection(
            args.port, baud=args.baud, source_system=245, source_component=191
        )
        print(f"Waiting for autopilot heartbeat on {args.port}...", flush=True)
        deadline = time.monotonic() + 15
        heartbeat = None
        while time.monotonic() < deadline:
            candidate = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if candidate is not None and candidate.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                heartbeat = candidate
                break
        if heartbeat is None:
            raise RuntimeError("No autopilot heartbeat within 15 seconds. Check port/power/port ownership.")

        system, component = heartbeat.get_srcSystem(), heartbeat.get_srcComponent()
        print(f"Connected: system={system}, component={component}")
        master.mav.command_long_send(
            system, component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR,
            int(1_000_000 / args.rate), 0, 0, 0, 0, 0
        )
        print(f"Requested DISTANCE_SENSOR at {args.rate} Hz. Ctrl+C stops.")
        print("Age = time since Pi received the message; Hz = received packets, not fresh sensor samples.")
        sensors = {}
        started = next_display = time.monotonic()
        next_flush = started + 1
        no_data_warning = False

        while True:
            msg = master.recv_match(blocking=True, timeout=0.1)
            received_ns = time.monotonic_ns()
            now = received_ns / 1_000_000_000
            if msg is not None and (msg.get_srcSystem(), msg.get_srcComponent()) == (system, component):
                kind = msg.get_type()
                if kind == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
                    result_enum = mavutil.mavlink.enums["MAV_RESULT"].get(msg.result)
                    result_name = result_enum.name if result_enum else str(msg.result)
                    print(f"Message-rate request: {result_name}", flush=True)
                elif kind == "DISTANCE_SENSOR":
                    key = (system, component, int(msg.id))
                    previous = sensors.get(key)
                    ticks = previous[2] if previous else deque(maxlen=1000)
                    ticks.append(now)
                    sensors[key] = (msg, now, ticks)
                    state = reading_status(msg)
                    if writer:
                        writer.writerow([
                            received_ns, msg.time_boot_ms, system, component,
                            msg.id, msg.orientation, msg.current_distance,
                            msg.current_distance / 100.0 if state == "IN RANGE" else "",
                            msg.min_distance, msg.max_distance, state
                        ])

            if now >= next_display:
                parts = []
                seen_orientations = set()
                for key, (sample, received, ticks) in sorted(sensors.items()):
                    while ticks and ticks[0] < now - 5:
                        ticks.popleft()
                    rate = len(ticks) / min(5.0, max(now - started, 0.001))
                    age = now - received
                    state = "STALE" if age > args.stale else reading_status(sample)
                    value = f"{sample.current_distance / 100.0:.2f} m" if state == "IN RANGE" else "--"
                    seen_orientations.add(sample.orientation)
                    parts.append(
                        f"{label(sample.orientation)} [id={key[2]}] {value} | {state} | "
                        f"raw={sample.current_distance} cm | age={age * 1000:.0f} ms | rx={rate:.1f} Hz"
                    )
                for orientation in (0, 25):
                    if orientation not in seen_orientations:
                        parts.append(f"{label(orientation)}: WAITING")
                print(" || ".join(parts), flush=True)
                next_display = now + 0.5

            if log and now >= next_flush:
                log.flush()
                next_flush = now + 1
            if not sensors and not no_data_warning and now - started >= 10:
                print("No DISTANCE_SENSOR received. Check request ACK, saved sensor parameters, "
                      "reboot, and whether another client overrides stream rates.", flush=True)
                no_data_warning = True
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if master is not None:
            master.close()
        if log is not None:
            log.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}")
