import serial

ser = serial.Serial("/dev/serial0", 57600, timeout=1)

ser.write(b"Hello\n")
print(ser.read(6))
