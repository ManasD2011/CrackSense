from pymavlink import mavutil

PORT = "/dev/serial0"
BAUD = 57600

print("Connecting to Pixhawk...")

master = mavutil.mavlink_connection(PORT, baud=BAUD)

print("Waiting for heartbeat (10 seconds)...")

heartbeat = master.wait_heartbeat(timeout=10)

if heartbeat:
    print("=================================")
    print("Connected successfully!")
    print(f"System ID    : {master.target_system}")
    print(f"Component ID : {master.target_component}")
    print("=================================")
else:
    print("No heartbeat received.")
