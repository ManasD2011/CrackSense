import json
from pathlib import Path
import struct
import sys
import unittest
for folder in ("shared", "raspberry-pi", "ground-station"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / folder))
from timing import ClockMap,History,decode,SCHEMA,NS
from rtsp_video import SenderClock,H264Assembler,rtp_packet,sender_reports,START
from telemetry_common import encode


def packet(seq=1,stamp=10*NS,orientation=0,sensor=0,session='a'*32):
    return dict(schema=SCHEMA,session_id=session,sequence=seq,system_id=1,component_id=1,
        sensor_id=sensor,orientation=orientation,raw_distance_cm=125,min_cm=10,max_cm=1200,
        signal_quality=0,status='IN_RANGE',distance_m=1.25,pixhawk_time_boot_ms=10000,
        pi_receive_monotonic_ns=10*NS,pi_receive_unix_ns=stamp+10_000_000,
        sample_unix_ns=stamp,clock_uncertainty_ns=2_000_000,clock_status='MAPPED_MESSAGE_TIME')

class TimingTests(unittest.TestCase):
    def test_clock_offset_and_expiry(self):
        c=ClockMap()
        for i in range(3):
            start=(20+i)*NS
            self.assertTrue(c.add(start,start+4_000_000,start+2_000_000-5*NS))
        estimate=c.estimate(23*NS)
        self.assertEqual(estimate[0],5*NS)
        self.assertGreaterEqual(estimate[1],3_000_000)
        self.assertIsNone(c.estimate(40*NS))
    def test_clock_reboot(self):
        c=ClockMap()
        for i in range(3): c.add((10+i)*NS,(10+i)*NS+2,5*NS+i*NS)
        c.add(13*NS,13*NS+2,NS)
        self.assertIsNone(c.estimate(13*NS+2))
    def test_validation(self):
        self.assertEqual(decode(encode(packet()))['distance_m'],1.25)
        for key,value in [('distance_m',float('nan')),('sequence',True),('sample_unix_ns',float('inf')),('clock_uncertainty_ns',-1)]:
            p=packet();p[key]=value
            with self.assertRaises(ValueError): decode(json.dumps(p).encode())
    def test_causal_matching_and_invalid(self):
        h=History()
        h.add(packet(1,10*NS),20*NS)
        h.add(packet(2,10*NS+100_000_000),20*NS)
        m=h.match(10*NS+50_000_000,1_000_000,0,0,20*NS)
        self.assertEqual(m['sequence'],1)
        self.assertEqual(m['sample_age_ms'],50)
        p=packet(3,10*NS+150_000_000)
        p.update(raw_distance_cm=0,distance_m=None,status='INVALID')
        h.add(p,20*NS)
        self.assertEqual(h.match(10*NS+160_000_000,0,0,0,20*NS)['status'],'INVALID')
    def test_stale_missing_and_uncertainty(self):
        h=History();h.add(packet(),20*NS)
        self.assertEqual(h.match(None,None,0,0,20*NS)['status'],'VIDEO_TIME_UNAVAILABLE')
        self.assertEqual(h.match(10*NS,0,0,0,22*NS)['status'],'TELEMETRY_STALE')
        self.assertEqual(h.match(10*NS+249_000_000,0,0,0,20*NS)['status'],'SAMPLE_STALE')
        self.assertEqual(h.match(10*NS-100_000_000,0,0,0,20*NS)['status'],'NO_PRIOR_SAMPLE')
    def test_duplicate_reorder_and_sessions(self):
        h=History();h.add(packet(2),20*NS)
        self.assertFalse(h.add(packet(1),20*NS))
        self.assertFalse(h.add(packet(2),20*NS))
        self.assertTrue(h.add(packet(1,session='b'*32),20*NS))
        self.assertFalse(h.add(packet(3),20*NS))
        self.assertEqual(len(h.samples),1)
    def test_sensor_ambiguity(self):
        h=History();h.add(packet(),20*NS)
        p=packet();p['system_id']=2;h.add(p,20*NS)
        self.assertEqual(h.match(10*NS,0,0,0,20*NS)['status'],'AMBIGUOUS_SENSOR')
    def test_mapping_loss_blocks_old_value(self):
        h=History();h.add(packet(),20*NS)
        p=packet(2);p.update(sample_unix_ns=None,clock_uncertainty_ns=None,clock_status='UNMAPPED')
        h.add(p,20*NS)
        self.assertEqual(h.match(10*NS,0,0,0,20*NS)['status'],'SENSOR_CLOCK_UNAVAILABLE')

class RTPTests(unittest.TestCase):
    def test_sender_report_wrap_expiry_jump(self):
        c=SenderClock();c.add(0xfffffff0,10*NS,0)
        self.assertEqual(c.map(0xfffffff0,0),(None,None))
        c.add((0xfffffff0+90000)&0xffffffff,11*NS,NS)
        self.assertEqual(c.map((0xfffffff0+180000)&0xffffffff,2*NS)[0],12*NS)
        self.assertEqual(c.map(1,20*NS),(None,None))
        c.add(180000,99*NS,3*NS)
        self.assertEqual(c.count,1)
    def test_rtcp_and_rtp(self):
        raw=struct.pack('!BBHIIIIII',128,200,6,123,2208988810,0,90000,1,1)
        self.assertEqual(list(sender_reports(raw)),[(123,90000,10*NS)])
        rtp=struct.pack('!BBHII',128,224,65535,4,123)+b'\x65abc'
        self.assertEqual(rtp_packet(rtp),(65535,4,123,True,96,b'\x65abc'))
    def test_fu_stap_and_sequence_wrap(self):
        a=H264Assembler()
        self.assertIsNone(a.push(65535,7,False,b'\x7c\x85ab'))
        self.assertEqual(a.push(0,7,True,b'\x7c\x45cd'),START+b'\x65abcd')
        stap=b'\x78\x00\x02\x67a\x00\x02\x68b'
        self.assertEqual(a.push(1,8,True,stap),START+b'\x67a'+START+b'\x68b')
        with self.assertRaises(ValueError): a.push(3,9,True,b'\x65x')
    def test_malformed_packets(self):
        for payload in (b'',b'\x80'*5):
            with self.assertRaises(ValueError):rtp_packet(payload)
        with self.assertRaises(ValueError): H264Assembler().push(1,1,True,b'\x78\x00\xffx')
        with self.assertRaises(ValueError): H264Assembler().push(1,1,True,b'\x7c\x41x')

if __name__=='__main__':unittest.main()
