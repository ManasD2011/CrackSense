"""
attitude_test.py

Test 2

Purpose:
    Read Roll, Pitch and Yaw from Pixhawk.

Outputs:
    Roll (deg)
    Pitch (deg)
    Yaw (deg)

Author: Ralkar
"""

from common import connect
from common import request_message
from common import banner

from pymavlink import mavutil

import math
import time


def main():

    banner("TEST 2 : ATTITUDE")

    master = connect()

    # Request ATTITUDE message at 10 Hz
    request_message(
        master,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        frequency=2
    )

    print("Receiving ATTITUDE messages...\n")

    previous_time = time.time()

    while True:

        msg = master.recv_match(
            type="ATTITUDE",
            blocking=True,
            timeout=2
        )

        if msg is None:
            print("No ATTITUDE message received.")
            continue

        current_time = time.time()

        dt = current_time - previous_time

        previous_time = current_time

        hz = 1.0 / dt if dt > 0 else 0

        roll = math.degrees(msg.roll)
        pitch = math.degrees(msg.pitch)
        yaw = math.degrees(msg.yaw)

        print("---------------------------------------------")

        print(f"Roll      : {roll:8.2f} deg")

        print(f"Pitch     : {pitch:8.2f} deg")

        print(f"Yaw       : {yaw:8.2f} deg")

        print(f"Rate      : {hz:6.2f} Hz")

        print()


if __name__ == "__main__":

    main()
