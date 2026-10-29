import time
from pymavlink import mavutil


# ============================================================
# CONFIGURATION
# ============================================================

PORT = "/dev/ttyACM0"
BAUD = 115200

# Test velocity command
#
# BODY_NED:
#   +VX = Forward
#   -VX = Backward
#   +VY = Right
#   -VY = Left
#   +VZ = Down
#   -VZ = Up
#
TEST_VX = 0.10
TEST_VY = 0.00
TEST_VZ = 0.00

SEND_RATE_HZ = 10
SEND_PERIOD = 1.0 / SEND_RATE_HZ


# ============================================================
# CONNECT TO PIXHAWK
# ============================================================

print("Connecting to Pixhawk...")

master = mavutil.mavlink_connection(
    PORT,
    baud=BAUD
)

print("Waiting for heartbeat...")

master.wait_heartbeat()

print()
print("============================================")
print("PIXHAWK CONNECTED")
print("============================================")

print(
    f"System ID     : {master.target_system}"
)

print(
    f"Component ID  : {master.target_component}"
)


# ============================================================
# SET GUIDED MODE
# ============================================================

print()
print("Setting GUIDED mode...")

modes = master.mode_mapping()

if modes is None:
    raise RuntimeError(
        "Could not get Pixhawk mode mapping"
    )

if "GUIDED" not in modes:
    raise RuntimeError(
        "GUIDED mode is not available"
    )

guided_mode_id = modes["GUIDED"]

master.set_mode(guided_mode_id)

time.sleep(1)


# ============================================================
# VERIFY GUIDED MODE
# ============================================================

heartbeat = master.recv_match(
    type="HEARTBEAT",
    blocking=True,
    timeout=2
)

if heartbeat is not None:

    current_mode = \
        mavutil.mode_string_v10(heartbeat)

    print(
        f"Current Pixhawk mode: {current_mode}"
    )

else:

    print(
        "Could not read current mode"
    )


# ============================================================
# REQUEST POSITION_TARGET_LOCAL_NED TELEMETRY
# ============================================================

print()
print(
    "Requesting POSITION_TARGET_LOCAL_NED..."
)

# MAVLink message ID:
#
# POSITION_TARGET_LOCAL_NED = 85
#
# 100000 microseconds = 10 Hz

master.mav.command_long_send(

    master.target_system,
    master.target_component,

    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,

    0,

    85,          # Message ID
    100000,      # Interval = 100 ms

    0,
    0,
    0,
    0,
    0
)

print(
    "Requested POSITION_TARGET_LOCAL_NED "
    "at 10 Hz"
)


# ============================================================
# VELOCITY COMMAND FUNCTION
# ============================================================

def send_velocity(vx, vy, vz):

    """
    Send velocity target to ArduPilot.

    Frame:
        MAV_FRAME_BODY_NED

    BODY_NED convention:

        +X = Forward
        -X = Backward

        +Y = Right
        -Y = Left

        +Z = Down
        -Z = Up

    Units:
        m/s
    """

    # --------------------------------------------------------
    # Ignore position
    # Ignore acceleration
    # Ignore yaw
    # Ignore yaw rate
    #
    # Keep velocity enabled.
    # --------------------------------------------------------

    type_mask = (

        mavutil.mavlink
        .POSITION_TARGET_TYPEMASK_X_IGNORE

        |

        mavutil.mavlink
        .POSITION_TARGET_TYPEMASK_Y_IGNORE

        |

        mavutil.mavlink
        .POSITION_TARGET_TYPEMASK_Z_IGNORE

        |

        mavutil.mavlink
        .POSITION_TARGET_TYPEMASK_AX_IGNORE

        |

        mavutil.mavlink
        .POSITION_TARGET_TYPEMASK_AY_IGNORE

        |

        mavutil.mavlink
        .POSITION_TARGET_TYPEMASK_AZ_IGNORE

        |

        mavutil.mavlink
        .POSITION_TARGET_TYPEMASK_YAW_IGNORE

        |

        mavutil.mavlink
        .POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )


    # --------------------------------------------------------
    # Send SET_POSITION_TARGET_LOCAL_NED
    # --------------------------------------------------------

    master.mav.set_position_target_local_ned_send(

        int(
            time.time() * 1000
        ) & 0xFFFFFFFF,

        master.target_system,
        master.target_component,

        # IMPORTANT:
        # BODY_NED = 8
        mavutil.mavlink.MAV_FRAME_BODY_NED,

        type_mask,

        # ----------------------------------------------------
        # Position
        # Ignored
        # ----------------------------------------------------

        0.0,
        0.0,
        0.0,

        # ----------------------------------------------------
        # Velocity
        # ----------------------------------------------------

        vx,
        vy,
        vz,

        # ----------------------------------------------------
        # Acceleration
        # Ignored
        # ----------------------------------------------------

        0.0,
        0.0,
        0.0,

        # ----------------------------------------------------
        # Yaw
        # Ignored
        # ----------------------------------------------------

        0.0,

        # ----------------------------------------------------
        # Yaw rate
        # Ignored
        # ----------------------------------------------------

        0.0
    )


# ============================================================
# READ PIXHAWK POSITION TARGET
# ============================================================

def read_pixhawk_target():

    """
    Read POSITION_TARGET_LOCAL_NED
    published by ArduPilot.
    """

    msg = master.recv_match(
        type="POSITION_TARGET_LOCAL_NED",
        blocking=False
    )

    return msg


# ============================================================
# MAIN TEST
# ============================================================

print()
print("============================================")
print("MAVLINK VELOCITY TEST")
print("============================================")

print()
print("Pi command:")
print(
    f"  Frame = "
    f"{mavutil.mavlink.MAV_FRAME_BODY_NED}"
)

print(
    f"  VX = {TEST_VX:+.2f} m/s"
)

print(
    f"  VY = {TEST_VY:+.2f} m/s"
)

print(
    f"  VZ = {TEST_VZ:+.2f} m/s"
)

print()
print("Expected BODY_NED:")
print("  +VX = Forward")
print("  +VY = Right")
print("  +VZ = Down")

print()
print("Press CTRL+C to stop.")
print()


last_status_time = 0.0


try:

    while True:

        loop_start = time.monotonic()


        # ====================================================
        # SEND COMMAND
        # ====================================================

        send_velocity(
            TEST_VX,
            TEST_VY,
            TEST_VZ
        )


        # ====================================================
        # PRINT WHAT PI SENT
        # ====================================================

        print(
            f"Pi sent:       "
            f"frame="
            f"{mavutil.mavlink.MAV_FRAME_BODY_NED} "
            f"vx={TEST_VX:+.2f} "
            f"vy={TEST_VY:+.2f} "
            f"vz={TEST_VZ:+.2f}"
        )


        # ====================================================
        # READ PIXHAWK MESSAGES
        # ====================================================

        while True:

            msg = master.recv_match(
                blocking=False
            )

            if msg is None:
                break


            msg_type = msg.get_type()


            # ------------------------------------------------
            # POSITION TARGET
            # ------------------------------------------------

            if msg_type == \
                    "POSITION_TARGET_LOCAL_NED":

                print(
                    f"Pixhawk target: "
                    f"frame={msg.coordinate_frame} "
                    f"vx={msg.vx:+.2f} "
                    f"vy={msg.vy:+.2f} "
                    f"vz={msg.vz:+.2f}"
                )


            # ------------------------------------------------
            # HEARTBEAT
            # ------------------------------------------------

            elif msg_type == "HEARTBEAT":

                now = time.monotonic()

                # Don't print heartbeat every loop
                if now - last_status_time > 1.0:

                    mode = \
                        mavutil.mode_string_v10(msg)

                    print(
                        f"Pixhawk mode: {mode}"
                    )

                    last_status_time = now


            # ------------------------------------------------
            # STATUS TEXT
            # ------------------------------------------------

            elif msg_type == "STATUSTEXT":

                print(
                    f"Pixhawk STATUS: "
                    f"{msg.text}"
                )


            # ------------------------------------------------
            # COMMAND ACK
            # ------------------------------------------------

            elif msg_type == "COMMAND_ACK":

                print(
                    f"COMMAND_ACK: "
                    f"command={msg.command} "
                    f"result={msg.result}"
                )


        # ====================================================
        # MAINTAIN SEND RATE
        # ====================================================

        elapsed = \
            time.monotonic() - loop_start

        sleep_time = \
            SEND_PERIOD - elapsed

        if sleep_time > 0:

            time.sleep(sleep_time)


except KeyboardInterrupt:

    print()
    print("Stopping...")


finally:

    # ========================================================
    # STOP VELOCITY
    # ========================================================

    print(
        "Sending zero velocity..."
    )

    for _ in range(10):

        send_velocity(
            0.0,
            0.0,
            0.0
        )

        time.sleep(0.05)


    print(
        "Test finished."
    )
