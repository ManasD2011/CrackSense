import serial
import time        

PORT = "/dev/ttyAMA0"   # try this instead of /dev/serial0 on Pi 5 — confirm with `ls -l /dev/serial0`
BAUD = 115200

def read_data(ser):
    print("Entering read_data()")
    while True:
        if ser.in_waiting >= 9:
            frame = ser.read(9)
            if frame[0] == 0x59 and frame[1] == 0x59:  # TFmini frame header
                distance = frame[2] + (frame[3] << 8)   # cm
                strength = frame[4] + (frame[5] << 8)
                print(f"Distance: {distance} cm | Strength: {strength}")
            else:
                # resync if header not found
                ser.read(1)
        else:
            time.sleep(0.01)

if __name__ == "__main__":
    try:
        print("Opening serial...")
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"Port open: {ser.is_open}")
        read_data(ser)
    except Exception as e:
        print("ERROR:", e)
