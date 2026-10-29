"""
common.py

Common MAVLink helper functions for all tests.

Author: Shubham
"""

from pymavlink import mavutil


# ============================================================
# Connect
# ============================================================

def connect(device="/dev/ttyACM0", baud=115200):

    print("Connecting to Pixhawk...")

    master = mavutil.mavlink_connection(
        device,
        baud=baud
    )

    master.wait_heartbeat()

    print("Heartbeat received.")

    print(f"System ID     : {master.target_system}")
    print(f"Component ID  : {master.target_component}")

    return master


# ============================================================
# Request MAVLink Message
# ============================================================

def request_message(master, message_id, frequency=10):

    interval = int(1e6 / frequency)

    master.mav.command_long_send(

        master.target_system,

        master.target_component,

        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,

        0,

        message_id,

        interval,

        0,

        0,

        0,

        0,

        0

    )


# ============================================================
# Print Banner
# ============================================================

def banner(title):

    print()

    print("=" * 50)

    print(title)

    print("=" * 50)

    print()
