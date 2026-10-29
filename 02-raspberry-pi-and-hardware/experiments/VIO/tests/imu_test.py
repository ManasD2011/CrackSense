"""
imu_test.py

Test 3

Purpose:
    Read HIGHRES_IMU messages from Pixhawk.

Outputs:
    Accelerometer
    Gyroscope
    Magnetometer
    Barometric Pressure
    Temperature

Author: Shubham
"""

from common import connect
from common import request_message
from common import banner

from pymavlink import mavutil

import time


def main():

    banner("TEST 3 : HIGHRES IMU")

    master = connect()

    # Request HIGHRES_IMU at 20 Hz
    request_message(
        master,
        mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU,
        frequency=3
    )

    print("Receiving HIGHRES_IMU...\n")

    previous_time = time.time()

    while True:

        msg = master.recv_match(
            type="HIGHRES_IMU",
            blocking=True,
            timeout=2
        )

        if msg is None:
            print("No IMU message received.")
            continue

        current_time = time.time()

        dt = current_time - previous_time

        previous_time = current_time

        hz = 1.0 / dt if dt > 0 else 0

        print("=" * 60)

        print(f"Rate : {hz:.1f} Hz")

        print()

        print("Accelerometer (m/s²)")
        print(f"  X : {msg.xacc:8.3f}")
        print(f"  Y : {msg.yacc:8.3f}")
        print(f"  Z : {msg.zacc:8.3f}")

        print()

        print("Gyroscope (rad/s)")
        print(f"  X : {msg.xgyro:8.3f}")
        print(f"  Y : {msg.ygyro:8.3f}")
        print(f"  Z : {msg.zgyro:8.3f}")

        print()

        print("Magnetometer (Gauss)")
        print(f"  X : {msg.xmag:8.3f}")
        print(f"  Y : {msg.ymag:8.3f}")
        print(f"  Z : {msg.zmag:8.3f}")

        print()

        print(f"Absolute Pressure : {msg.abs_pressure:.2f} mbar")
        print(f"Differential Press: {msg.diff_pressure:.2f} mbar")
        print(f"Temperature       : {msg.temperature/100:.2f} °C")

        print()


if __name__ == "__main__":

    main()
