"""
local_position_test.py

Test 5

Purpose:
    Read EKF Local Position (LOCAL_POSITION_NED)

Outputs:
    Position (X, Y, Z)
    Velocity (VX, VY, VZ)

Author: Shubham
"""

from common import connect
from common import request_message
from common import banner

from pymavlink import mavutil

import time


def main():

    banner("TEST 5 : LOCAL POSITION NED")

    master = connect()

    # Request LOCAL_POSITION_NED at 10 Hz
    request_message(
        master,
        mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
        frequency=10
    )

    print("Receiving LOCAL_POSITION_NED...\n")

    previous_time = time.time()

    while True:

        msg = master.recv_match(
            type="LOCAL_POSITION_NED",
            blocking=True,
            timeout=2
        )

        if msg is None:
            print("No LOCAL_POSITION_NED received.")
            continue

        current_time = time.time()

        dt = current_time - previous_time
        previous_time = current_time

        hz = 1.0 / dt if dt > 0 else 0

        print("=" * 60)

        print(f"Rate : {hz:.2f} Hz")
        print()

        print("Position (meters)")
        print(f"X : {msg.x:8.3f}")
        print(f"Y : {msg.y:8.3f}")
        print(f"Z : {msg.z:8.3f}")

        print()

        print("Velocity (m/s)")
        print(f"VX : {msg.vx:8.3f}")
        print(f"VY : {msg.vy:8.3f}")
        print(f"VZ : {msg.vz:8.3f}")

        print()


if __name__ == "__main__":

    main()
