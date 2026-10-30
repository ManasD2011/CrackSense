import serial

ser = serial.Serial('/dev/serial0',57600,timeout=2)

data = ser.read(100)

print(len(data))
print(data)
