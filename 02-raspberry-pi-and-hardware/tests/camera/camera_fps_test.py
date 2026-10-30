from picamera2 import Picamera2
import time
import cv2

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

# Let camera exposure/auto-gain settle
time.sleep(2)

frame_count = 0
start_time = time.perf_counter()

print("Starting camera FPS test...")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        frame_start = time.perf_counter()

        frame = picam2.capture_array()

        # Convert to grayscale for optical-flow processing
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame_count += 1

        frame_time = time.perf_counter() - frame_start

        elapsed = time.perf_counter() - start_time

        if elapsed >= 2.0:
            fps = frame_count / elapsed

            print(
                f"Frames: {frame_count:5d} | "
                f"FPS: {fps:6.2f} | "
                f"Frame time: {frame_time*1000:6.2f} ms | "
                f"Resolution: {gray.shape[1]}x{gray.shape[0]}"
            )

            frame_count = 0
            start_time = time.perf_counter()

except KeyboardInterrupt:
    print("\nTest stopped.")

finally:
    picam2.stop()
