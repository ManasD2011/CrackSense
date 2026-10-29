"""
heartbeat_test.py

Test 1

Checks:
1. USB connection
2. MAVLink connection
3. Heartbeat reception

Author: Shubham
"""

from common import connect
from common import banner

import time


def main():

    banner("TEST 1 : HEARTBEAT")

    start = time.time()

    master = connect()

    end = time.time()

    print()

    print("Connection Successful")

    print(f"Time to Connect : {end-start:.2f} sec")

    print()

    print("Waiting for 5 Heartbeats...\n")

    heartbeat_count = 0

    while heartbeat_count < 5:

        msg = master.recv_match(
            type="HEARTBEAT",
            blocking=True
        )

        if msg is None:
            continue

        heartbeat_count += 1

        print(f"Heartbeat {heartbeat_count}")

        print(f"Base Mode      : {msg.base_mode}")

        print(f"Custom Mode    : {msg.custom_mode}")

        print(f"System Status  : {msg.system_status}")

        print(f"MAV Type       : {msg.type}")

        print(f"Autopilot      : {msg.autopilot}")

        print("-" * 40)

    print()

    print("TEST PASSED")

    print("Heartbeat communication is working.")


if __name__ == "__main__":

    main()
