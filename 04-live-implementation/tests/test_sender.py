"""Exercise the real sender loop with a deterministic MAVLink transport double."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
for folder in ("shared", "raspberry-pi", "ground-station"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
import pi_telemetry_sender as sender
from timing import decode,NS

class SenderTest(unittest.TestCase):
    def test_requests_timesync_and_emits_mapped_two_sensor_packets(self):
        class Clock:
            now=20*NS
            def monotonic_ns(self):return self.now
            def monotonic(self):return self.now/NS
            def time_ns(self):return self.now+1_700_000_000*NS
            def sleep(self,_):raise AssertionError('Unexpected reconnect')
        clock=Clock()
        class Message(SimpleNamespace):
            def get_srcSystem(self):return 1
            def get_srcComponent(self):return 1
            def get_type(self):return self.kind
        class Master:
            def __init__(self):
                self.mav=self;self.pending=[];self.count=0;self.closed=False;self.requests=[]
            def command_long_send(self,*args):self.requests.append(args)
            def timesync_send(self,tc1,ts1):
                self.pending.append(Message(kind='TIMESYNC',tc1=ts1+1_000_000-5*NS,ts1=ts1))
            def recv_match(self,**kwargs):
                if kwargs.get('type')=='HEARTBEAT':return Message(kind='HEARTBEAT',autopilot=3)
                if self.pending:
                    clock.now+=2_000_000
                    return self.pending.pop(0)
                self.count+=1;clock.now+=100_000_000
                if self.count>90:raise KeyboardInterrupt()
                if self.count%5==0:return Message(kind='HEARTBEAT',autopilot=3)
                i=self.count%2
                return Message(kind='DISTANCE_SENSOR',id=i,orientation=25 if i else 0,
                    current_distance=125,min_distance=10,max_distance=1200,signal_quality=0,
                    time_boot_ms=(clock.now-5*NS)//1_000_000)
            def close(self):self.closed=True
        master=Master()
        class UDP:
            def __init__(self):self.packets=[]
            def setblocking(self,_):pass
            def sendto(self,data,dest):self.packets.append(decode(data))
            def close(self):pass
        udp=UDP()
        mavutil=SimpleNamespace(mavlink_connection=lambda *a,**kw:master,
            mavlink=SimpleNamespace(MAV_AUTOPILOT_INVALID=8))
        fake_socket=SimpleNamespace(AF_INET=2,SOCK_DGRAM=2,socket=lambda *a:udp,gethostbyname=lambda x:x)
        with patch.dict(sys.modules,{'pymavlink':SimpleNamespace(mavutil=mavutil)}), \
             patch.object(sender,'time',clock),patch.object(sender,'socket',fake_socket), \
             patch.object(sys,'argv',['pi_telemetry_sender.py','--host','127.0.0.1']),contextlib.redirect_stdout(io.StringIO()):
            sender.main()
        self.assertTrue(master.closed)
        self.assertEqual(master.requests[0][2:5],(511,0,132))
        self.assertEqual({p['orientation'] for p in udp.packets},{0,25})
        mapped=[p for p in udp.packets if p['sample_unix_ns'] is not None]
        self.assertGreater(len(mapped),20)
        for p in mapped:
            self.assertLess(abs(p['sample_unix_ns']-p['pi_receive_unix_ns']),2_000_000)
            self.assertEqual(p['clock_status'],'MAPPED_MESSAGE_TIME')

if __name__=='__main__':unittest.main()
