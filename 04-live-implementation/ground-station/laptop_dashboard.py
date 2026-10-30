"""Timestamp probe and paired OpenCV dashboard. Run --probe before live inference."""
import sys
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1] / "shared"))

import argparse
from collections import deque
import json
from pathlib import Path
import queue
import socket
import threading
import time
import uuid
from timing import History, decode, NS
from rtsp_video import RTSP


class Runtime:
    def __init__(self,args):
        self.args=args
        self.lock=threading.Lock()
        self.stop=threading.Event()
        self.history=History()
        self.latest=None
        self.result=None
        self.generation=0
        self.sequence=0
        self.state='CONNECTING'
        self.error=''
        self.fatal=None
        self.reconnects=0
        self.bad_packets=0
        self.source_ip=args.source_ip
        self.last_packet_error=''
        self.threads=[]
        self.udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            self.udp.bind((args.bind,args.udp_port))
            self.udp.settimeout(.2)
        except Exception:
            self.udp.close()
            raise RuntimeError('Cannot bind telemetry port. Stop laptop_telemetry_receiver.py first.')

    def start(self,predict):
        for fn in (self.telemetry,self.video,lambda:self.infer(predict)):
            t=threading.Thread(target=fn,daemon=True)
            self.threads.append(t); t.start()

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=6)
        self.udp.close()

    def telemetry(self):
        while not self.stop.is_set():
            try:
                data,address=self.udp.recvfrom(2048)
                received=time.monotonic_ns()
                if self.source_ip and address[0]!=self.source_ip:
                    continue
                p=decode(data)
                with self.lock:
                    if self.source_ip is None:
                        self.source_ip=address[0]
                    self.history.add(p,received)
            except socket.timeout:
                continue
            except (ValueError,KeyError,TypeError) as exc:
                with self.lock:
                    self.bad_packets+=1
                    self.last_packet_error=str(exc)
            except OSError as exc:
                if not self.stop.is_set():
                    self.fatal=f'Telemetry socket: {exc}'; self.stop.set()

    def video(self):
        while not self.stop.is_set():
            client=None
            with self.lock:
                self.generation+=1
                self.latest=self.result=None
                self.state='CONNECTING'
            try:
                client=RTSP(self.args.stream)
                client.start()
                for frame in client.frames(self.stop):
                    with self.lock:
                        self.sequence+=1
                        frame.update(generation=self.generation,frame_id=self.sequence)
                        self.latest=frame
                        self.state='LIVE'; self.error=''
            except Exception as exc:
                with self.lock:
                    self.state='DISCONNECTED'
                    self.error=str(exc)
                    self.latest=self.result=None
                    self.reconnects+=1
                print('Video:',exc,flush=True)
            finally:
                if client is not None:
                    client.close()
            self.stop.wait(.5)

    def infer(self,predict):
        last_id=0
        last_generation=-1
        try:
            while not self.stop.is_set():
                with self.lock:
                    frame=self.latest
                    if frame is None or frame['frame_id']==last_id:
                        frame=None
                    elif time.monotonic_ns()-frame['decoded_ns']>self.args.stale_seconds*NS:
                        frame=None
                    if frame is not None:
                        now=time.monotonic_ns()
                        common=(frame['source_unix_ns'],frame['video_uncertainty_ns'])
                        matches={
                            'ground':self.history.match(*common,25,self.args.ground_id,now,self.args.max_sample_age_ms),
                            'wall':self.history.match(*common,0,self.args.wall_id,now,self.args.max_sample_age_ms)}
                        telemetry_session=self.history.session
                if frame is None:
                    self.stop.wait(.002); continue
                skipped=max(0,frame['frame_id']-last_id-1) if last_generation==frame['generation'] else 0
                last_id=frame['frame_id']; last_generation=frame['generation']
                begin=time.monotonic_ns()
                mask=None if predict is None else predict(frame['image'])
                done=time.monotonic_ns()
                result=dict(frame,matches=matches,mask=mask,inference_start_ns=begin,
                            result_ready_ns=done,processing_ms=None if predict is None else (done-begin)/1e6,
                            skipped_waiting_frames=skipped,telemetry_session=telemetry_session)
                with self.lock:
                    if frame['generation']==self.generation and telemetry_session==self.history.session:
                        self.result=result
        except Exception as exc:
            self.fatal=f'Inference failed: {exc}'; self.stop.set()


class Output:
    """Bounded nonblocking queue; recording/snapshot writes never block inference."""
    def __init__(self,path,args):
        self.path=path
        self.args=args
        self.queue=queue.Queue(maxsize=16)
        self.dropped=0
        self.recorded=0
        self.error=None
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def put(self,item):
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped+=1
            return False

    def run(self):
        writers={}
        sizes=None
        part=0
        try:
            with (self.path/'frames.jsonl').open('w',encoding='utf-8') as log:
                flushed=time.monotonic()
                record_index=0
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        kind,payload=self.queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if kind=='result':
                        row,canvas,raw,mask=payload
                        if canvas is not None:
                            import cv2
                            images={'raw':raw,'processed':make_overlay(raw,mask),'paired':canvas}
                            new_sizes={name:(img.shape[1],img.shape[0]) for name,img in images.items()}
                            if new_sizes!=sizes:
                                for writer in writers.values():
                                    writer.release()
                                writers={}
                                part+=1
                                record_index=0
                                suffix='' if part==1 else f'_part{part:03d}'
                                names={name:f'{name}{suffix}.avi' for name in images}
                                for name,size in new_sizes.items():
                                    writer=cv2.VideoWriter(str(self.path/names[name]),cv2.VideoWriter_fourcc(*'MJPG'),
                                        self.args.record_fps,size)
                                    writers[name]=writer
                                    if not writer.isOpened():
                                        raise IOError(f'Could not open {names[name]}')
                                sizes=new_sizes
                            for name,img in images.items():
                                writers[name].write(img)
                            row=dict(row,recorded_frame_index=record_index,recording_part=part,
                                     recording_files=names,recording_sizes=sizes)
                            record_index+=1
                            self.recorded+=1
                        log.write(json.dumps(row,allow_nan=False)+'\n')
                        if time.monotonic()-flushed>1:
                            log.flush(); flushed=time.monotonic()
                    elif kind=='snapshot':
                        import cv2
                        frame,mask,canvas,row=payload
                        stem=f"g{row['generation']}_f{row['frame_id']}_{time.time_ns()}"
                        for suffix,img in [('raw',frame),('dashboard',canvas),('mask',None if mask is None else mask*255)]:
                            if img is not None and not cv2.imwrite(str(self.path/f'{stem}_{suffix}.png'),img):
                                raise IOError('Snapshot write failed')
                        (self.path/f'{stem}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
                log.flush()
        except Exception as exc:
            self.error=str(exc)
        finally:
            for writer in writers.values():
                writer.release()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.error='Output worker did not finish within 10 seconds'


def metadata(result,now,received_fps,result_fps,args):
    excluded={'image','mask'}
    row={k:v for k,v in result.items() if k not in excluded}
    row.update(model=Path(args.checkpoint).name if not args.preview_only else 'preview',
               precision=args.precision.upper() if not args.preview_only else 'NONE',
               received_fps=received_fps,displayed_result_fps=result_fps,
               decode_to_display_preparation_ms=(now-result['decoded_ns'])/1e6,
               queue_wait_ms=(result['inference_start_ns']-result['decoded_ns'])/1e6,
               timing_status='APPROXIMATE_MESSAGE_TIME',
               capture_to_screen_latency_ms=None)
    return row


def make_overlay(raw,mask):
    """Full-resolution prediction overlay; never modify the raw source array."""
    import numpy as np
    overlay=raw.copy()
    if mask is not None:
        pixels=mask.astype(bool)
        overlay[pixels]=(overlay[pixels]*.55+np.array([0,0,255])*.45).astype(np.uint8)
    return overlay


def render(result,row,state='LIVE'):
    import cv2
    import numpy as np
    canvas=np.full((620,1280,3),22,np.uint8)
    def text(s,x,y,color=(225,225,225),scale=.52):
        cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    text('RAW',15,25,scale=.65)
    text('PROCESSED',655,25,scale=.65)
    if result is None:
        text(state,30,100,(0,200,255),.7)
        text('Waiting for a fresh paired result',655,100)
        return canvas
    raw=result['image']
    overlay=make_overlay(raw,result['mask'])
    for x,img in ((0,raw),(640,overlay)):
        ratio=min(640/img.shape[1],360/img.shape[0])
        w,h=round(img.shape[1]*ratio),round(img.shape[0]*ratio)
        resized=cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
        y=40+(360-h)//2
        canvas[y:y+h,x+(640-w)//2:x+(640-w)//2+w]=resized
    text(f"Frame {row['frame_id']} | session {row['generation']}",160,25)
    text(f"Frame {row['frame_id']} | session {row['generation']}",800,25)
    ms=row['processing_ms']
    lines=[f"Model: {row['model'][:44]} | {row['precision']}",
           'Processing: --' if ms is None else f'Processing: {ms:.1f} ms',
           f"Received: {row['received_fps']:.1f} FPS | Displayed: {row['displayed_result_fps']:.1f} FPS"]
    for label,key in [('Ground range','ground'),('Wall range','wall')]:
        m=row['matches'][key]
        if m['distance_m'] is None:
            lines.append(f"{label}: -- ({m['status']})")
        else:
            lines.append(f"{label}: {m['distance_m']:.2f} m | sample age {m['sample_age_ms']:.1f} ms")
    lines.extend(['Timing: APPROXIMATE message-time association',
                  'Q: quit | S: save paired snapshot'])
    for index,line in enumerate(lines):
        text(line,655,428+index*26, (0,205,255) if index==5 else (225,225,225))
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream',required=True)
    p.add_argument('--source-ip',help='Restrict telemetry to Pi IPv4; otherwise first valid sender is locked')
    p.add_argument('--bind',default='0.0.0.0')
    p.add_argument('--udp-port',type=int,default=5005)
    p.add_argument('--ground-id',type=int,default=1)
    p.add_argument('--wall-id',type=int,default=0)
    p.add_argument('--checkpoint',default='Quantised/latest_fp16.pt')
    p.add_argument('--precision',choices=['fp16','fp32'],default='fp16')
    p.add_argument('--threshold',type=float)
    p.add_argument('--max-sample-age-ms',type=float,default=250)
    p.add_argument('--stale-seconds',type=float,default=1)
    p.add_argument('--duration',type=float,default=0)
    p.add_argument('--probe',action='store_true',help='No GUI/model. Default 30 seconds. Run this first.')
    p.add_argument('--preview-only',action='store_true')
    p.add_argument('--headless',action='store_true',help='No window; diagnostics/automated test only')
    p.add_argument('--record',action='store_true',help='Save full-resolution raw/processed plus paired AVI; default is metrics only')
    p.add_argument('--record-fps',type=float,default=25)
    p.add_argument('--output',default='phase4_results')
    args=p.parse_args()
    import math
    if not all(math.isfinite(v) for v in (args.duration,args.max_sample_age_ms,args.stale_seconds,args.record_fps)):
        p.error('Timing values must be finite')
    if args.duration<0 or not 0<args.max_sample_age_ms<=1000 or not 0<args.stale_seconds<=5 or not 0<args.record_fps<=120:
        p.error('Invalid timing parameters')
    if not 1<=args.udp_port<=65535 or not all(0<=i<=255 for i in (args.ground_id,args.wall_id)):
        p.error('Invalid UDP port or sensor ID')
    if args.threshold is not None and not 0<=args.threshold<=1:
        p.error('Threshold must be finite and in [0,1]')
    if args.probe:
        args.headless=args.preview_only=True
        args.record=False
        args.duration=args.duration or 30
    import av  # Fail early with installation guidance in README.
    runtime=Runtime(args)
    predict=None
    threshold=None
    try:
        if not args.preview_only:
            from model import load_predictor
            predict,threshold=load_predictor(args)
    except Exception:
        runtime.close()
        raise
    path=Path(args.output)/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    path.mkdir(parents=True)
    (path/'config.json').write_text(json.dumps(dict(vars(args),used_threshold=threshold,
        pyav=av.__version__,timing='RTP/RTCP source time vs mapped MAVLink message time; approximate'),indent=2),encoding='utf-8')
    output=Output(path,args)
    runtime.start(predict)
    started=rate_start=time.monotonic()
    received_before=shown_before=shown=0
    received_fps=result_fps=0
    last_id=-1
    mapped=matched=0
    processing=deque(maxlen=10000)
    console_at=0
    row=current=canvas=None
    try:
        if not args.headless:
            import cv2
            cv2.namedWindow('CrackSense',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('CrackSense',1280,620)
        while not runtime.stop.is_set():
            now=time.monotonic_ns()
            with runtime.lock:
                current=runtime.result
                generation=runtime.generation
                telemetry_session=runtime.history.session
                state=runtime.state
                received=runtime.sequence
                accepted=runtime.history.accepted
                last_error=runtime.last_packet_error
            if time.monotonic()-rate_start>=1:
                elapsed=time.monotonic()-rate_start
                received_fps=(received-received_before)/elapsed
                result_fps=(shown-shown_before)/elapsed
                received_before,shown_before,rate_start=received,shown,time.monotonic()
            valid=(current is not None and state=='LIVE' and current['generation']==generation
                   and current['telemetry_session']==telemetry_session
                   and now-current['decoded_ns']<=args.stale_seconds*NS)
            if valid:
                new=current['frame_id']!=last_id
                if new:
                    last_id=current['frame_id']; shown+=1
                    row=metadata(current,now,received_fps,result_fps,args)
                    mapped+=current['source_unix_ns'] is not None
                    matched+=all(m['status']=='MATCHED_APPROXIMATE' for m in row['matches'].values())
                    if row['processing_ms'] is not None:
                        processing.append(row['processing_ms'])
                    canvas=render(current,row) if not args.headless or args.record else None
                    output.put(('result',(row,canvas if args.record else None,
                                          current['image'] if args.record else None,
                                          current['mask'] if args.record else None)))
            else:
                canvas=render(None,None,state+' / WAITING OR STALE') if not args.headless else None
                row=None
            if time.monotonic()>console_at:
                statuses='waiting' if row is None else ', '.join(f"{k}={v['status']}" for k,v in row['matches'].items())
                print(f'video={state} received={received} mapped={mapped}/{shown} '
                      f'telemetry={accepted} both_matched={matched}/{shown} {statuses}'
                      +(f' | packet_error={last_error}' if last_error else ''),flush=True)
                console_at=time.monotonic()+1
            if output.error:
                raise IOError(output.error)
            if not args.headless:
                cv2.imshow('CrackSense',canvas)
                key=cv2.waitKey(1)&255
                if key==ord('q') or cv2.getWindowProperty('CrackSense',cv2.WND_PROP_VISIBLE)<1:
                    break
                if key==ord('s') and valid and row is not None:
                    ok=output.put(('snapshot',(current['image'],current['mask'],canvas,row)))
                    print('Snapshot queued' if ok else 'Snapshot not saved: output queue full')
            else:
                runtime.stop.wait(.003)
            if args.duration and time.monotonic()-started>=args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close(); output.close()
        if not args.headless:
            cv2.destroyAllWindows()
        summary=dict(received_frames=runtime.sequence,displayed_or_probed_results=shown,
            video_time_mapped_results=mapped,both_ranges_matched_results=matched,
            accepted_telemetry=runtime.history.accepted,rejected_telemetry=runtime.bad_packets+runtime.history.rejected,
            telemetry_sequence_gaps=runtime.history.gaps,reconnects=runtime.reconnects,
            output_dropped_items=output.dropped,recorded_frame_sets=output.recorded,
            error=runtime.fatal or output.error,
            last_video_error=runtime.error,last_packet_error=runtime.last_packet_error,
            elapsed_s=time.monotonic()-started,physical_timing_validated=False,
            gate='MAPPING_OBSERVED_REQUIRES_HARDWARE_VALIDATION' if matched>=10 else 'NOT_READY',
            processing_last_10000_count=len(processing),
            processing_mean_ms=sum(processing)/len(processing) if processing else None,
            processing_p95_ms=sorted(processing)[min(len(processing)-1,int(.95*len(processing)))] if processing else None)
        (path/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('Results:',path)
        print(json.dumps(summary,indent=2))
    if runtime.fatal or output.error:
        raise RuntimeError(runtime.fatal or output.error)
    if args.probe and matched<10:
        raise SystemExit(2)

if __name__=='__main__':
    main()
