# Historical Phase2

Pi MediaMTX streaming notes; original configuration snippets have formatting errors. Use the current raspberry-pi/phase4.yml instead.

These original notes contain old working directories, local addresses and commands. See [current startup](../../docs/startup-sequence.md).

<details>
<summary>Original project notes</summary>

Phase2:

This phase is on the raspberry pi 5

**Step 1: Download MediaMTX on the Pi**

Commands:

mkdir /phase2

cd /phase2

wget -O mediamtxv1.21.1linuxarm64.tar.gz [https://github.com/bluenviron/mediamtx/releases/download/v1.21.1/mediamtxv1.21.1linuxarm64.tar.gz](https://github.com/bluenviron/mediamtx/releases/download/v1.21.1/mediamtx_v1.21.1_linux_arm64.tar.gz)

tar -xzf mediamtxv1.21.1linuxarm64.tar.gz

**-------------------------------------------------------------------------------**

**Step 2: Create a separate streaming configuration**

cat > phase2.yml <<'YAML'

logLevel: info

rtsp: yes

rtspTransports: tcp]

rtspAddress: :8554

paths:

  cam0:

```
source: rpiCamera

rpiCameraCamID: 0

rpiCameraWidth: 1280

rpiCameraHeight: 720

rpiCameraFPS: 25

rpiCameraCodec: softwareH264

rpiCameraBitrate: 4000000

rpiCameraIDRPeriod: 25

rpiCameraH264Profile: baseline
```

YAML

---

**Step 3: Start streaming, Close any other process using camera 0, then run:**

Command:

./mediamtx phase2.yml

---

**Step 4: Open the stream on your Windows laptop**

command:

"C:Program FilesVideoLANVLCvlc.exe" --rtsp-tcp --network-caching=100 rtsp://10.207.215.185:8554/cam0

VLC has a delay. Use FFPlay

Download the FFplay in your laptop (Mine in E:Drive)

Open the extracted folder, then its bin folder. You should see ffplay.exe.

Click File Explorer’s address bar, type cmd, and press Enter. This opens CMD directly in that folder.

Command:

ffplay.exe -version

ffplay.exe -rtsp_transport tcp -fflags nobuffer -flags low_delay -framedrop -noinfbuf -an rtsp://10.207.215.185:8554/cam0

**Step5: run virtual environment from the root directory**

D:\Web development\Crack_Detection_Project>

**Commands:**

.venv\Scripts\Activate.ps1

cd B2_Exp

python .\Live_Implementation\Phase3\laptop_live_[inference.py](http://inference.py) --stream rtsp://10.207.215.185:8554/cam0 --checkpoint Quantised/latest_[fp16.pt](http://fp16.pt) --duration 900

</details>
