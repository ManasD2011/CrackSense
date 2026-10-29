from pymavlink import mavutil
import time

master = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

master.wait_heartbeat()

print("Connected")

# Request ATTITUDE at 10 Hz
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
    0,
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
    100000,      # microseconds = 10 Hz
    0,
    0,
    0,
    0,
    0
)

while True:

    msg = master.recv_match(
        type="ATTITUDE",
        blocking=True,
        timeout=2
    )

    print(msg)