from flask import Flask, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)

picam2 = Picamera2(1)

config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()


def generate_frames():
    while True:
        frame = picam2.capture_array()

        # RGB -> JPEG
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>Raspberry Pi Camera</title>
    </head>
    <body>
        <h2>IMX219 Live Feed</h2>
        <img src="/video" width="640" height="480">
    </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        threaded=True
    )
