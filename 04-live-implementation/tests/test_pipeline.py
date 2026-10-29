"""Local TCP RTSP/H264 + UDP integration, using real PyAV encode/decode.
No Pi, GPU or camera is simulated as validated hardware.
"""
from fractions import Fraction
import json
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
for folder in ("shared", "raspberry-pi", "ground-station"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
from test_core import packet
from telemetry_common import encode
from timing import NS

class PipelineTest(unittest.TestCase):
    def test_real_decoder_telemetry_recording_and_reconnect(self):
        try:
            import av
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest('Install av numpy opencv-python to run integration test')
        encoder=av.CodecContext.create('libx264','w')
        encoder.width=320;encoder.height=180;encoder.pix_fmt='yuv420p'
        encoder.time_base=Fraction(1,25);encoder.framerate=Fraction(25,1)
        encoder.options={'preset':'ultrafast','tune':'zerolatency','profile':'baseline',
                         'x264-params':'keyint=10:repeat-headers=1:scenecut=0'}
        units=[]
        for i in range(80):
            img=np.zeros((180,320,3),np.uint8)
            img[:,:,1]=50+i
            cv2.putText(img,str(i),(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
            frame=av.VideoFrame.from_ndarray(img,format='bgr24');frame.pts=i
            for pkt in encoder.encode(frame):units.append(bytes(pkt))
        units.extend(bytes(p) for p in encoder.encode(None))
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen()
        listener.settimeout(.2)
        port=listener.getsockname()[1]
        reserve=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);reserve.bind(('127.0.0.1',0))
        udp_port=reserve.getsockname()[1];reserve.close()
        stop=threading.Event(); errors=[];sessions=[]

        def serve():
            udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            sequences=[0,0]
            connection_index=0
            try:
                while not stop.is_set():
                    try: conn,_=listener.accept()
                    except socket.timeout:continue
                    connection_index+=1;sessions.append(connection_index)
                    conn.settimeout(2)
                    url=f'rtsp://127.0.0.1:{port}/cam0'
                    with conn:
                        buf=b''
                        while True:
                            while b'\r\n\r\n' not in buf:
                                data=conn.recv(4096)
                                if not data:raise EOFError()
                                buf+=data
                            header,buf=buf.split(b'\r\n\r\n',1)
                            lines=header.decode().split('\r\n')
                            method=lines[0].split()[0]
                            cseq=next(x.split(':',1)[1].strip() for x in lines if x.lower().startswith('cseq:'))
                            body=b''
                            extra=''
                            if method=='DESCRIBE':
                                body=b'v=0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\na=fmtp:96 packetization-mode=1\r\na=control:trackID=0\r\n'
                                extra=f'Content-Base: {url}/\r\nContent-Type: application/sdp\r\n'
                            if method=='SETUP':extra='Session: test\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n'
                            conn.sendall(f'RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\n{extra}Content-Length: {len(body)}\r\n\r\n'.encode()+body)
                            if method=='PLAY':break
                        seq=0;ssrc=123+connection_index
                        base=time.time_ns()-80_000_000
                        start=time.monotonic()
                        def media(ch,data):conn.sendall(bytes([36,ch])+len(data).to_bytes(2,'big')+data)
                        count=18 if connection_index==1 else 80
                        for index,unit in enumerate(units[:count]):
                            if stop.is_set():break
                            stamp=(0xffff0000+index*3600)&0xffffffff
                            unix=base+index*40_000_000
                            for sensor,orientation in [(0,0),(1,25)]:
                                sequences[sensor]+=1
                                p=packet(sequences[sensor],unix-10_000_000,orientation,sensor)
                                udp.sendto(encode(p),('127.0.0.1',udp_port))
                            if index%5==0:
                                seconds,fraction=divmod(unix,NS)
                                report=struct.pack('!BBHIIIIII',128,200,6,ssrc,seconds+2208988800,
                                                   fraction*2**32//NS,stamp,1,1)
                                media(1,report)
                            nals=[n for n in re.split(b'\x00\x00\x00?\x01',unit) if n]
                            payloads=[]
                            for nal in nals:
                                if len(nal)<=400:payloads.append(nal)
                                else:
                                    chunks=[nal[i:i+398] for i in range(1,len(nal),398)]
                                    for j,chunk in enumerate(chunks):
                                        fu=(nal[0]&31)|(128 if j==0 else 0)|(64 if j==len(chunks)-1 else 0)
                                        payloads.append(bytes([(nal[0]&224)|28,fu])+chunk)
                            for j,payload in enumerate(payloads):
                                seq=(seq+1)&65535
                                media(0,struct.pack('!BBHII',128,96|(128 if j==len(payloads)-1 else 0),seq,stamp,ssrc)+payload)
                            delay=start+(index+1)*.04-time.monotonic()
                            if delay>0:stop.wait(delay)
            # Windows can report the deliberate reconnect/teardown as WSAECONNABORTED.
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
            except Exception as exc:errors.append(repr(exc))
            finally:udp.close()
        worker=threading.Thread(target=serve,daemon=True);worker.start()
        try:
            with tempfile.TemporaryDirectory() as output:
                command=[sys.executable,str(Path(__file__).resolve().parents[1]/'ground-station'/'laptop_dashboard.py'),
                    '--stream',f'rtsp://127.0.0.1:{port}/cam0','--udp-port',str(udp_port),
                    '--source-ip','127.0.0.1','--preview-only','--headless','--record',
                    '--duration','3','--output',output]
                run=subprocess.run(command,capture_output=True,text=True,timeout=15)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                session=next(Path(output).iterdir())
                summary=json.loads((session/'summary.json').read_text())
                self.assertGreater(summary['received_frames'],20,run.stdout)
                self.assertGreater(summary['both_ranges_matched_results'],10,run.stdout)
                self.assertGreaterEqual(summary['reconnects'],1)
                rows=[json.loads(x) for x in (session/'frames.jsonl').read_text().splitlines()]
                self.assertGreaterEqual(len({r['generation'] for r in rows}),2)
                for row in rows:
                    for match in row['matches'].values():
                        if match['distance_m'] is not None:
                            self.assertGreaterEqual(row['source_unix_ns'],match['sample_unix_ns'])
                            self.assertLess(match['sample_age_ms'],250)
                cap=cv2.VideoCapture(str(session/'paired.avi'));ok,frame=cap.read();cap.release()
                self.assertTrue(ok)
                self.assertEqual(frame.shape,(620,1280,3))
                self.assertFalse(errors,errors)
                print('Integration:',summary['received_frames'],'decoded;',summary['both_ranges_matched_results'],
                      'matched;',summary['reconnects'],'reconnect(s)')
        finally:
            stop.set();worker.join(timeout=3);listener.close()

if __name__=='__main__':unittest.main()
