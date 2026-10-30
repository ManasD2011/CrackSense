"""Narrow RTSP/TCP H264 receiver retaining RTP -> RTCP absolute timestamps.
For this project's unauthenticated MediaMTX baseline H264 stream only.
Supports single NAL, STAP-A, FU-A; rejects other modes. PyAV handles decoding.
"""
import sys
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1] / "shared"))

import base64
from fractions import Fraction
import re
import socket
import struct
import time
from urllib.parse import urlsplit

NS=1_000_000_000
START=b'\x00\x00\x00\x01'
RECEIVER_REVISION='phase4-h264-headers-2'


class PicturePackets:
    """Keep standalone codec headers until a picture can be sent to FFmpeg.

    An RTP marker is not proof that its payload contains a picture. In
    particular, separate SPS/PPS/SEI units must not be decoded as pictures.
    Preserve their bytes and associate only the following picture's RTP time.
    """
    def __init__(self):
        self.pending=bytearray()
        self.seen_key=False

    def prepare(self,unit):
        nals=[n for n in unit.split(START)[1:] if n]
        types=[n[0]&31 for n in nals]
        if not any(1<=kind<=5 for kind in types):
            self.pending.extend(unit)
            if len(self.pending)>1024*1024:
                raise ValueError('Too many H264 headers without a picture')
            return None
        if not self.seen_key and 5 not in types:
            # Retain any in-band parameter sets while waiting for an IDR.
            for nal in nals:
                if (nal[0]&31) in (7,8):
                    self.pending.extend(START+nal)
            if len(self.pending)>1024*1024:
                raise ValueError('Too many H264 parameters before an IDR')
            return None
        self.seen_key=True
        picture=bytes(self.pending)+unit
        self.pending.clear()
        return picture

def signed_delta(a,b):
    return ((a-b+2**31)%2**32)-2**31

class SenderClock:
    def __init__(self):
        self.reference=None
        self.count=0
        self.residual=0

    def add(self,rtp,unix_ns,now):
        if self.reference:
            old_rtp,old_unix,_=self.reference
            residual=abs(unix_ns-(old_unix+signed_delta(rtp,old_rtp)*NS//90000))
            self.residual=residual
            if residual>20_000_000:
                self.count=0
        self.reference=(rtp,unix_ns,now)
        self.count+=1

    def map(self,rtp,now):
        if self.reference is None or self.count<2:
            return None,None
        ref,unix,received=self.reference
        if now-received>15*NS or abs(signed_delta(rtp,ref))>15*90000:
            return None,None
        return unix+signed_delta(rtp,ref)*NS//90000,self.residual+1_000_000


def sender_reports(data):
    offset=0
    while offset<len(data):
        if len(data)-offset<4:
            raise ValueError('Truncated RTCP')
        flags,kind,length=struct.unpack_from('!BBH',data,offset)
        size=(length+1)*4
        if flags>>6!=2 or offset+size>len(data):
            raise ValueError('Invalid RTCP')
        if kind==200:
            if size<28:
                raise ValueError('Short sender report')
            ssrc,seconds,fraction,rtp=struct.unpack_from('!IIII',data,offset+4)
            yield ssrc,rtp,(seconds-2208988800)*NS+(fraction*NS>>32)
        offset+=size


def rtp_packet(data):
    if len(data)<12 or data[0]>>6!=2:
        raise ValueError('Invalid RTP')
    flags,marker_pt,sequence,timestamp,ssrc=struct.unpack_from('!BBHII',data)
    offset=12+4*(flags&15)
    if offset>len(data):
        raise ValueError('Truncated CSRC')
    if flags&16:
        if offset+4>len(data):
            raise ValueError('Truncated extension')
        offset+=4+4*struct.unpack_from('!H',data,offset+2)[0]
    end=len(data)
    if flags&32:
        pad=data[-1]
        if pad==0 or pad>end-offset:
            raise ValueError('Invalid padding')
        end-=pad
    if offset>=end:
        raise ValueError('Empty payload')
    return sequence,timestamp,ssrc,bool(marker_pt&128),marker_pt&127,data[offset:end]


class H264Assembler:
    def __init__(self):
        self.timestamp=None
        self.sequence=None
        self.nals=[]
        self.fragment=None
        self.size=0

    def push(self,seq,stamp,marker,payload):
        if self.sequence is not None and seq!=(self.sequence+1)%65536:
            raise ValueError('RTP sequence gap: reconnect to a fresh decoder')
        self.sequence=seq
        if self.timestamp!=stamp:
            if self.nals or self.fragment is not None:
                raise ValueError('Incomplete H264 access unit')
            self.timestamp=stamp
            self.size=0
        self.size+=len(payload)
        if self.size>8*1024*1024:
            raise ValueError('Oversized H264 access unit')
        kind=payload[0]&31
        if 1<=kind<=23:
            if self.fragment is not None:
                raise ValueError('Interrupted fragment')
            self.nals.append(START+payload)
        elif kind==24:
            pos=1
            while pos<len(payload):
                if pos+2>len(payload):
                    raise ValueError('Short STAP-A')
                n=int.from_bytes(payload[pos:pos+2],'big'); pos+=2
                if n==0 or pos+n>len(payload):
                    raise ValueError('Invalid STAP-A length')
                self.nals.append(START+payload[pos:pos+n]); pos+=n
        elif kind==28:
            if len(payload)<3:
                raise ValueError('Short FU-A')
            fu=payload[1]
            if fu&128:
                if self.fragment is not None:
                    raise ValueError('Nested FU-A')
                self.fragment=bytearray(START+bytes([(payload[0]&224)|(fu&31)]))
            if self.fragment is None:
                # Joining mid-access-unit: reconnect until an intact unit arrives.
                raise ValueError('FU-A continuation without start')
            self.fragment.extend(payload[2:])
            if fu&64:
                self.nals.append(bytes(self.fragment)); self.fragment=None
        else:
            raise ValueError(f'Unsupported H264 RTP packet type {kind}')
        if marker:
            if self.fragment is not None:
                raise ValueError('Incomplete FU-A')
            result=b''.join(self.nals)
            self.nals=[]
            self.timestamp=None
            return result
        return None


class RTSP:
    def __init__(self,url):
        parts=urlsplit(url)
        if parts.scheme!='rtsp' or not parts.hostname or parts.username or parts.password:
            raise ValueError('Expected unauthenticated rtsp://host:port/path')
        self.url=url
        self.socket=socket.create_connection((parts.hostname,parts.port or 554),timeout=3)
        self.socket.settimeout(2)
        self.buffer=bytearray()
        self.cseq=0
        self.session=None
        self.payload_type=None
        self.parameters=b''
        self.rtp_channel=0
        self.rtcp_channel=1

    def close(self):
        self.socket.close()

    def fill(self,n):
        while len(self.buffer)<n:
            data=self.socket.recv(65536)
            if not data:
                raise EOFError('RTSP connection closed')
            self.buffer.extend(data)
            if len(self.buffer)>16*1024*1024:
                raise ValueError('RTSP buffer limit exceeded')

    def event(self):
        self.fill(1)
        if self.buffer[0]==36:
            self.fill(4)
            channel=self.buffer[1]
            size=int.from_bytes(self.buffer[2:4],'big')
            self.fill(4+size)
            body=bytes(self.buffer[4:4+size]); del self.buffer[:4+size]
            return 'media',channel,body
        while b'\r\n\r\n' not in self.buffer:
            self.fill(len(self.buffer)+1)
            if len(self.buffer)>65536:
                raise ValueError('Oversized RTSP header')
        end=self.buffer.index(b'\r\n\r\n')+4
        lines=bytes(self.buffer[:end]).decode('utf-8').split('\r\n')
        if not lines[0].startswith('RTSP/1.0 '):
            raise ValueError('Unexpected RTSP message')
        headers={}
        for line in lines[1:]:
            if ':' in line:
                k,v=line.split(':',1); headers[k.lower()]=v.strip()
        size=int(headers.get('content-length','0'))
        if not 0<=size<=1024*1024:
            raise ValueError('Oversized RTSP body')
        self.fill(end+size)
        body=bytes(self.buffer[end:end+size]); del self.buffer[:end+size]
        return 'response',(int(lines[0].split()[1]),headers),body

    def send(self,method,url=None,extra=None):
        self.cseq+=1
        headers={'CSeq':str(self.cseq),'User-Agent':'CrackSense-Phase4'}
        if self.session:
            headers['Session']=self.session
        headers.update(extra or {})
        request=f'{method} {url or self.url} RTSP/1.0\r\n'+''.join(f'{k}: {v}\r\n' for k,v in headers.items())+'\r\n'
        self.socket.sendall(request.encode())

    def request(self,method,url=None,extra=None):
        self.send(method,url,extra)
        while True:
            kind,meta,body=self.event()
            if kind=='response':
                status,headers=meta
                if status!=200:
                    raise RuntimeError(f'{method}: RTSP status {status}')
                return headers,body

    def start(self):
        headers,sdp=self.request('DESCRIBE',extra={'Accept':'application/sdp'})
        video=False
        control=None
        fmtp={}
        for line in sdp.decode().splitlines():
            if line.startswith('m='):
                if video:
                    break
                video=line.startswith('m=video ')
            if not video:
                continue
            match=re.match(r'a=rtpmap:(\d+) H264/90000',line,re.I)
            if match:
                self.payload_type=int(match[1])
            if line.startswith('a=control:'):
                control=line.split(':',1)[1]
            if line.startswith('a=fmtp:'):
                pt,params=line[7:].split(' ',1)
                fmtp[int(pt)]=dict(x.strip().split('=',1) for x in params.split(';') if '=' in x)
        if self.payload_type is None or control is None:
            raise ValueError('No H264/90000 video track in SDP')
        params=fmtp.get(self.payload_type,{})
        if params.get('packetization-mode','0') not in ('0','1'):
            raise ValueError('Unsupported H264 packetization mode')
        if 'sprop-parameter-sets' in params:
            self.parameters=b''.join(START+base64.b64decode(x,validate=True) for x in params['sprop-parameter-sets'].split(','))
        base=headers.get('content-base',self.url+'/')
        if control.startswith('rtsp://'):
            track=control
        elif control.startswith('/'):
            u=urlsplit(base); track=f'{u.scheme}://{u.netloc}{control}'
        else:
            track=base.rstrip('/')+'/'+control
        headers,_=self.request('SETUP',track,{'Transport':'RTP/AVP/TCP;unicast;interleaved=0-1'})
        self.session=headers['session'].split(';',1)[0]
        match=re.search(r'interleaved=(\d+)-(\d+)',headers.get('transport',''))
        if not match:
            raise ValueError('Server did not negotiate TCP interleaving')
        self.rtp_channel,self.rtcp_channel=map(int,match.groups())
        self.request('PLAY')

    def frames(self,stop):
        import av
        decoder=av.CodecContext.create('h264','r')
        decoder.thread_count=1
        decoder.extradata=self.parameters
        assembler=H264Assembler()
        clock=SenderClock()
        ssrc=None
        pictures=PicturePackets()
        last_rtp=None
        unwrapped=0
        keepalive=time.monotonic()+15
        last_frame=time.monotonic()
        while not stop.is_set():
            if time.monotonic()>keepalive:
                self.send('OPTIONS'); keepalive=time.monotonic()+15
            if time.monotonic()-last_frame>5:
                raise TimeoutError('No decoded frame for 5 seconds')
            kind,meta,data=self.event()
            if kind=='response':
                if meta[0]!=200:
                    raise RuntimeError(f'Keepalive RTSP status {meta[0]}')
                continue
            if meta==self.rtcp_channel:
                for report_ssrc,stamp,unix in sender_reports(data):
                    if ssrc is None or report_ssrc==ssrc:
                        if ssrc is None:
                            ssrc=report_ssrc
                        clock.add(stamp,unix,time.monotonic_ns())
                continue
            if meta!=self.rtp_channel:
                continue
            seq,stamp,packet_ssrc,marker,pt,payload=rtp_packet(data)
            if pt!=self.payload_type:
                continue
            if ssrc is not None and packet_ssrc!=ssrc:
                raise ValueError('Video SSRC changed')
            ssrc=packet_ssrc
            unit=assembler.push(seq,stamp,marker,payload)
            if unit is None:
                continue
            unit=pictures.prepare(unit)
            if unit is None:
                continue
            if last_rtp is None:
                unwrapped=stamp
            else:
                step=signed_delta(stamp,last_rtp)
                if step<0 or step>90000*5:
                    raise ValueError('Unexpected H264 baseline timestamp ordering')
                unwrapped+=step
            last_rtp=stamp
            packet=av.Packet(unit)
            packet.pts=packet.dts=unwrapped
            packet.time_base=Fraction(1,90000)
            try:
                decoded_frames=decoder.decode(packet)
            except av.error.InvalidDataError as exc:
                types=[n[0]&31 for n in unit.split(START)[1:] if n]
                raise ValueError(f'{RECEIVER_REVISION}: H264 picture decode failed; '
                                 f'rtp={stamp}, nal_types={types}, bytes={len(unit)}: {exc}') from exc
            for frame in decoded_frames:
                if frame.pts is None:
                    raise ValueError('Decoder discarded frame PTS')
                pts=int(Fraction(frame.pts)*frame.time_base*90000)
                now=time.monotonic_ns()
                source_ns,uncertainty=clock.map(pts&0xffffffff,now)
                last_frame=time.monotonic()
                yield dict(image=frame.to_ndarray(format='bgr24'),decoded_ns=now,
                           source_unix_ns=source_ns,video_uncertainty_ns=uncertainty,
                           rtp_timestamp=pts&0xffffffff,ssrc=ssrc,
                           sender_reports=clock.count)
