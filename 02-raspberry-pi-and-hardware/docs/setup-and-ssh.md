# Pi setup and SSH

Use Raspberry Pi OS with the supported camera stack. Enable SSH and configure the Pi and laptop on the intended network. From the laptop, replace the placeholders with your account and address:

```bash
ssh YOUR_PI_USER@YOUR_PI_IP
```

Clone this repository on the Pi or copy the phase folders while preserving their directory structure. On the Pi, from the repository root:

```bash
sudo apt update
sudo apt install python3-venv python3-picamera2 python3-opencv
python3 -m venv --system-site-packages .venv-pi
source .venv-pi/bin/activate
python -m pip install -r 02-raspberry-pi-and-hardware/requirements.txt
```

The system-site-packages option exposes OS camera bindings to the virtual environment. Do not install a second pip OpenCV distribution into this environment unless you deliberately replace the OS build. ArUco experiments require a build containing `cv2.aruco`; verify that separately.

Check the OS camera installation using [Raspberry Pi camera documentation](https://www.raspberrypi.com/documentation/accessories/camera-software.html). Use [MediaMTX installation guidance](https://mediamtx.org/docs/kickoff/install) for the Linux ARM64 build when you reach live streaming. The saved project configuration must be supported by the installed MediaMTX version.

SSH is remote shell access. It is not the transport for the final video stream or telemetry packets. Do not publish passwords, private keys, local account configuration or Kaggle credentials. If USB serial access is denied, check device permissions and the Pi account's serial-access group before retrying.
