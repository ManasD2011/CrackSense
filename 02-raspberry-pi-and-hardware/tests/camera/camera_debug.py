import cv2

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

print("Opened:", cap.isOpened())

if cap.isOpened():
	ret, frame = cap.read()

	print("Read:", ret)

	if ret:
    		print("Shape:", frame.shape)

cap.release()
