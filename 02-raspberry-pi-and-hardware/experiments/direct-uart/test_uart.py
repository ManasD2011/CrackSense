import serial

ser = serial.Serial('/dev/serial0', 57600, timeout=1)

message = b"Hello UART!\n"

ser.write(message)

received = ser.read(len(message))

print("Sent    :", message)
print("Received:", received)

ser.close()
