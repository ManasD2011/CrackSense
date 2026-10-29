from pymavlink import mavutil
import time

PORT = "/dev/ttyACM0"
BAUD = 115200

print("Connecting to Pixhawk...")

connection = mavutil.mavlink_connection(
    PORT,
    baud=BAUD
)

print("Waiting for heartbeat...")
connection.wait_heartbeat()

print("Connected!")
print(f"System ID: {connection.target_system}")
print(f"Component ID: {connection.target_component}")
print("----------------------------------------")

# Request DISTANCE_SENSOR message at 10 Hz
connection.mav.command_long_send(
    connection.target_system,
    connection.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0,
    132,       # DISTANCE_SENSOR message ID
    100000,    # 100000 us = 10 Hz
    0, 0, 0, 0, 0
)

print("Requested DISTANCE_SENSOR at 10 Hz")
print("----------------------------------------")

while True:

    msg = connection.recv_match(
        type="DISTANCE_SENSOR",
        blocking=True,
        timeout=2
    )

    if msg is None:
        print("No DISTANCE_SENSOR received")
        continue

    distance_cm = msg.current_distance

    print(
        f"LiDAR Distance: {distance_cm} cm "
        f"({distance_cm / 100:.2f} m)"
    )