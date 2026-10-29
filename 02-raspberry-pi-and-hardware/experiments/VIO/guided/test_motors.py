import time
from pymavlink import mavutil


# ============================================================
# CONFIG
# ============================================================

PORT = "/dev/ttyACM0"
BAUD = 115200

TEST_VX = 0.10       # BODY_NED: + = forward
TEST_VY = 0.00       # + = right
TEST_VZ = 0.00       # + = down

SEND_RATE = 10
SEND_PERIOD = 1.0 / SEND_RATE


# ============================================================
# CONNECT
# ============================================================

print("Connecting to Pixhawk...")

master = mavutil.mavlink_connection(
    PORT,
    baud=BAUD
)

print("Waiting for heartbeat...")

master.wait_heartbeat()

print()
print("Connected to Pixhawk")
print(
    f"System={master.target_system}, "
    f"Component={master.target_component}"
)


# ============================================================
# SET GUIDED
# ============================================================

print()
print("Setting GUIDED...")

modes = master.mode_mapping()

if modes is None:
    raise RuntimeError("Could not get mode mapping")

if "GUIDED" not in modes:
    raise RuntimeError("GUIDED mode not available")

master.set_mode(
    modes["GUIDED"]
)

time.sleep(1)


# ============================================================
# CHECK MODE
# ============================================================

hb = master.recv_match(
    type="HEARTBEAT",
    blocking=True,
    timeout=2
)

if hb:

    mode = mavutil.mode_string_v10(hb)

    print(
        f"Current mode: {mode}"
    )


# ============================================================
# VELOCITY COMMAND
# ============================================================

def send_velocity(vx, vy, vz):

    """
    MAV_FRAME_BODY_NED

        +vx = Forward
        -vx = Backward

        +vy = Right
        -vy = Left

        +vz = Down
        -vz = Up
    """

    type_mask = (

        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    )

    master.mav.set_position_target_local_ned_send(

        int(time.time() * 1000) & 0xFFFFFFFF,

        master.target_system,
        master.target_component,

        mavutil.mavlink.MAV_FRAME_BODY_NED,

        type_mask,

        # Position
        0,
        0,
        0,

        # Velocity
        vx,
        vy,
        vz,

        # Acceleration
        0,
        0,
        0,

        # Yaw
        0,

        # Yaw rate
        0
    )


# ============================================================
# READ MOTOR OUTPUTS
# ============================================================

def read_motor_outputs():

    """
    Read SERVO_OUTPUT_RAW from Pixhawk.

    servo1_raw ... servo4_raw
    correspond to the first four
    ArduPilot output channels.
    """

    msg = master.recv_match(
        type="SERVO_OUTPUT_RAW",
        blocking=False
    )

    if msg is None:
        return None

    return (
        msg.servo1_raw,
        msg.servo2_raw,
        msg.servo3_raw,
        msg.servo4_raw
    )


# ============================================================
# TEST
# ============================================================

print()
print("==============================================")
print("MOTOR RESPONSE TEST")
print("==============================================")

print(
    f"Command: "
    f"VX={TEST_VX:+.2f} "
    f"VY={TEST_VY:+.2f} "
    f"VZ={TEST_VZ:+.2f}"
)

print()
print("BODY_NED:")
print("  +VX = Forward")
print("  +VY = Right")
print("  +VZ = Down")

print()
print("PROPELLERS MUST BE REMOVED.")
print("==============================================")
print()


try:

    while True:

        # ----------------------------------------------------
        # Send velocity command
        # ----------------------------------------------------

        send_velocity(
            TEST_VX,
            TEST_VY,
            TEST_VZ
        )


        print(
            f"Command sent: "
            f"vx={TEST_VX:+.2f} "
            f"vy={TEST_VY:+.2f} "
            f"vz={TEST_VZ:+.2f}"
        )


        # ----------------------------------------------------
        # Read motor outputs
        # ----------------------------------------------------

        outputs = read_motor_outputs()

        if outputs:

            m1, m2, m3, m4 = outputs

            print(
                f"Motor outputs: "
                f"M1={m1} "
                f"M2={m2} "
                f"M3={m3} "
                f"M4={m4}"
            )

        else:

            print(
                "Motor outputs: "
                "no SERVO_OUTPUT_RAW received"
            )


        print("----------------------------------------------")

        time.sleep(SEND_PERIOD)


except KeyboardInterrupt:

    print()
    print("Stopping...")


finally:

    # --------------------------------------------------------
    # STOP COMMAND
    # --------------------------------------------------------

    print("Sending zero velocity...")

    for _ in range(20):

        send_velocity(
            0.0,
            0.0,
            0.0
        )

        time.sleep(0.05)

    print("Test finished.")
