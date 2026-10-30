from pymavlink import mavutil
import time

PORT = "/dev/ttyACM0"
BAUD = 115200

master = mavutil.mavlink_connection(PORT, baud=BAUD)

print("Waiting for Pixhawk...")
master.wait_heartbeat()

print("Connected")
print("System:", master.target_system)

def set_mode(mode):
    prev_mode = mode
    if mode not in master.mode_mapping():
        print("Unknown mode:", mode)
        print("Available modes:")
        print(list(master.mode_mapping().keys()))
        return

    mode_id = master.mode_mapping()[mode]

    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )

    time.sleep(1)

    msg = master.recv_match(
        type='HEARTBEAT',
        blocking=True
    )

    current_mode = mavutil.mode_string_v10(msg)

    print("Requested:", prev_mode)
    print("Current:", current_mode)


# =========================
# SELECT MODE
# =========================

set_mode("ALT_HOLD")
