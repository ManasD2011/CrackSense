#!/usr/bin/env python3
"""Receive lidar UDP packets. Python standard library only; no pymavlink needed.

Age means time since laptop reception, NOT sensor age at camera exposure.
This program does not synchronize clocks or measure one-way network latency.
"""
import argparse
import json
import socket
import time
from collections import deque
from telemetry_common import decode, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--source-ip", help="Optional filter: actual Pi IPv4 address")
    parser.add_argument("--stale", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 = until Ctrl+C")
    parser.add_argument("--log", help="Optional new JSONL file; never overwrites")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 < args.stale <= 60 or not 0 <= args.duration < 86400:
        parser.error("Invalid port, timeout or duration")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    log = None
    try:
        sock.bind((args.bind, args.port))
        sock.settimeout(0.1)
        if args.log:
            log = open(args.log, "x", encoding="utf-8")
        print(f"Listening on {args.bind}:{args.port}; waiting for Pi...", flush=True)
        print("Age = laptop receive age. Telemetry is NOT yet synchronized with video.", flush=True)
        source_ip = args.source_ip
        session = None
        retired_sessions = deque(maxlen=32)
        sensors = {}
        accepted = rejected = old = gaps = 0
        started = next_display = time.monotonic()
        next_flush = started + 1
        while not args.duration or time.monotonic()-started < args.duration:
            try:
                data, source = sock.recvfrom(2048)
            except socket.timeout:
                data = None
            received_ns = time.monotonic_ns()
            now = received_ns / 1e9
            if data is not None:
                try:
                    if source_ip is not None and source[0] != source_ip:
                        raise ValueError("unexpected sender")
                    p = decode(data)
                    if source_ip is None:
                        source_ip = source[0]
                        print(f"Using sender {source_ip}; restart receiver to change sender IP.", flush=True)
                    if p["session_id"] in retired_sessions:
                        raise ValueError("retired session")
                    if p["session_id"] != session:
                        if session is not None:
                            retired_sessions.append(session)
                        session = p["session_id"]
                        sensors.clear()
                        print(f"New sender session {session[:8]}; cleared previous readings.", flush=True)
                    key = (p["system_id"], p["component_id"], p["sensor_id"])
                    previous = sensors.get(key)
                    if previous and p["sequence"] <= previous[0]["sequence"]:
                        old += 1
                    else:
                        if previous:
                            gaps += max(0, p["sequence"]-previous[0]["sequence"]-1)
                        elif len(sensors) >= 32:
                            raise ValueError("too many sensors")
                        ticks = previous[2] if previous else deque(maxlen=1000)
                        ticks.append(now)
                        sensors[key] = (p, now, ticks)
                        accepted += 1
                        if log:
                            log.write(json.dumps({"laptop_receive_monotonic_ns": received_ns,
                                                  "source_ip": source[0], "packet": p}, allow_nan=False)+"\n")
                except (ValueError, KeyError, TypeError, UnicodeError):
                    rejected += 1
            if now >= next_display:
                parts, orientations = [], set()
                for key, (p, received, ticks) in sorted(sensors.items()):
                    while ticks and ticks[0] < now-5:
                        ticks.popleft()
                    hz = len(ticks)/min(5.0, max(now-started, 0.001))
                    age = now-received
                    state = "STALE" if age > args.stale else p["status"]
                    value = f'{p["distance_m"]:.2f} m' if state == "IN_RANGE" else "--"
                    orientations.add(p["orientation"])
                    parts.append(f'{label(p["orientation"])} [id={key[2]}] {value} | {state} | age={age*1000:.0f} ms | rx={hz:.1f} Hz')
                for orientation in (0, 25):
                    if orientation not in orientations:
                        parts.append(f"{label(orientation)}: WAITING")
                print(" || ".join(parts)+f" || received={accepted} rejected={rejected} old={old} gaps={gaps}", flush=True)
                next_display = now+0.5
            if log and now >= next_flush:
                log.flush()
                next_flush = now+1
    except KeyboardInterrupt:
        print("\nStopped receiver.")
    finally:
        if log is not None:
            log.close()
        sock.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Receiver error: {exc}")
