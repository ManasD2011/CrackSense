from pymavlink import mavutil

master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
master.wait_heartbeat()

print("Connected")

count = {}

while True:

    msg = master.recv_match(blocking=True)

    if msg is None:
        continue

    name = msg.get_type()

    count[name] = count.get(name, 0) + 1

    if count[name] == 1:
        print(name)
