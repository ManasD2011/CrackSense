from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)

print("Waiting for heartbeat...")

master.wait_heartbeat()

print("Heartbeat received!")

print("System ID:", master.target_system)
print("Component ID:", master.target_component)
