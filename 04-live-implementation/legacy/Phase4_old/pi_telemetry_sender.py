#!/usr/bin/env python3
"""Forward Pixhawk DISTANCE_SENSOR packets to the laptop; telemetry requests only.

Run instead of heartbeat_test.py / lidar_mavlink_test.py on the same serial port.
No arming, mode changes, motion commands or parameter writes are sent.
Restart this program if the Pixhawk USB connection is unplugged or rebooted.
"""
import argparse
import socket
import time
import uuid
from telemetry_common import make_packet, encode, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="10.207.215.7")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--device", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--rate", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.udp_port <= 65535 or not 1 <= args.rate <= 50 or args.baud <= 0:
        parser.error("Invalid UDP port, baud or rate (rate must be 1..50 Hz)")
    from pymavlink import mavutil

    master = None
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setblocking(False)
    try:
        destination = (socket.gethostbyname(args.host), args.udp_port)
        master = mavutil.mavlink_connection(args.device, baud=args.baud,
                                          source_system=245, source_component=191)
        print(f"Waiting for autopilot on {args.device}...", flush=True)
        deadline = time.monotonic() + 15
        heartbeat = None
        while time.monotonic() < deadline:
            h = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if h is not None and h.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                heartbeat = h
                break
        if heartbeat is None:
            raise RuntimeError("No autopilot heartbeat in 15 seconds")
        system, component = heartbeat.get_srcSystem(), heartbeat.get_srcComponent()
        master.mav.command_long_send(
            system, component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR,
            int(1_000_000 / args.rate), 0, 0, 0, 0, 0
        )
        session = uuid.uuid4().hex
        print(f"Connected {system}/{component}; sending to {destination[0]}:{destination[1]}", flush=True)
        print("UDP send success is not a laptop delivery acknowledgement. Ctrl+C stops.", flush=True)
        sequences, last = {}, {}
        sent = errors = 0
        next_report = time.monotonic()
        while True:
            msg = master.recv_match(blocking=True, timeout=0.1)
            received_ns = time.monotonic_ns()
            now = received_ns / 1e9
            if msg is not None and (msg.get_srcSystem(), msg.get_srcComponent()) == (system, component):
                kind = msg.get_type()
                if kind == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
                    entry = mavutil.mavlink.enums["MAV_RESULT"].get(msg.result)
                    print("Rate request:", entry.name if entry else msg.result, flush=True)
                elif kind == "DISTANCE_SENSOR":
                    key = int(msg.id)
                    sequences[key] = sequences.get(key, 0) + 1
                    packet = make_packet(msg, session, sequences[key], received_ns)
                    last[key] = (packet, now)
                    try:
                        udp.sendto(encode(packet), destination)
                        sent += 1
                    except OSError as exc:
                        errors += 1
                        if errors == 1 or errors % 100 == 0:
                            print(f"UDP send error: {exc}", flush=True)
            if now >= next_report:
                parts = []
                for key, (p, received) in sorted(last.items()):
                    state = p["status"] if now-received <= 1 else "STALE"
                    value = f'{p["distance_m"]:.2f} m' if state == "IN_RANGE" else "--"
                    parts.append(f'{label(p["orientation"])} id={key}: {value} {state}')
                print(f'Sent={sent}, send_errors={errors} | '+(" | ".join(parts) or "WAITING FOR LIDAR"), flush=True)
                next_report = now + 1
    except KeyboardInterrupt:
        print("\nStopped sender.")
    finally:
        if master is not None:
            master.close()
        udp.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Sender error: {exc}")
