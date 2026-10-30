"""
altitude_test.py

Test 4

Purpose:
    Read altitude information from Pixhawk.

Reads:
    GLOBAL_POSITION_INT

Outputs:
    Relative Altitude
    Absolute Altitude
    Ground Speed

Author: Shubham
"""

from common import connect
from common import request_message
from common import banner

from pymavlink import mavutil

import time


def main():

    banner("TEST 4 : ALTITUDE")

    master = connect()

    # Request GLOBAL_POSITION_INT at 10 Hz
    request_message(
        master,
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        frequency=10
    )

    print("Receiving GLOBAL_POSITION_INT...\n")

    previous_time = time.time()

    while True:

        msg = master.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=2
        )

        if msg is None:
            print("No GLOBAL_POSITION_INT received.")
            continue

        current_time = time.time()

        dt = current_time - previous_time
        previous_time = current_time

        hz = 1.0 / dt if dt > 0 else 0

        # Convert units
        relative_alt = msg.relative_alt / 1000.0      # mm → m
        absolute_alt = msg.alt / 1000.0               # mm → m
        vx = msg.vx / 100.0                           # cm/s → m/s
        vy = msg.vy / 100.0
        vz = msg.vz / 100.0

        ground_speed = (vx**2 + vy**2) ** 0.5

        print("=" * 60)

        print(f"Rate               : {hz:.2f} Hz")
        print()
        print(f"Relative Altitude  : {relative_alt:.3f} m")
        print(f"Absolute Altitude  : {absolute_alt:.3f} m")
        print()
        print(f"Velocity X         : {vx:.3f} m/s")
        print(f"Velocity Y         : {vy:.3f} m/s")
        print(f"Velocity Z         : {vz:.3f} m/s")
        print()
        print(f"Ground Speed       : {ground_speed:.3f} m/s")
        print()


if __name__ == "__main__":
    main()
