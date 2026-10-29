from common import connect

master = connect()

while True:

    msg = master.recv_match(blocking=True)

    if msg:

        print(msg.get_type())
