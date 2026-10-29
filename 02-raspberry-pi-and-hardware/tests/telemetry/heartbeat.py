from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)

master.wait_heartbeat()

print("Connected")

while True:
    msg = master.recv_match(blocking=True)

    if msg is not None:
        print(msg.get_type())
