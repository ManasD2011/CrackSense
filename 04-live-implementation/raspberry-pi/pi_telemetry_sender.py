"""Sole Pixhawk reader: DISTANCE_SENSOR + TIMESYNC -> versioned UDP packets."""
import sys
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1] / "shared"))

import argparse
from collections import defaultdict
import socket
import time
import uuid
from telemetry_common import make_packet, encode, label
from timing import ClockMap, SCHEMA, NS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',required=True,help='Ground-station laptop IPv4 address')
    parser.add_argument('--udp-port',type=int,default=5005)
    parser.add_argument('--device',default='/dev/ttyACM0')
    parser.add_argument('--baud',type=int,default=115200)
    parser.add_argument('--rate',type=float,default=10)
    args=parser.parse_args()
    if not 0<args.rate<=50 or not 1<=args.udp_port<=65535:
        parser.error('Use a rate in (0,50] and a valid UDP port')
    from pymavlink import mavutil
    udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    udp.setblocking(False)
    destination=(socket.gethostbyname(args.host),args.udp_port)
    print('Stop other programs reading',args.device)
    master=None
    try:
        while True:
            try:
                master=mavutil.mavlink_connection(args.device,baud=args.baud,
                    source_system=245,source_component=191)
                print('Waiting for autopilot heartbeat...')
                deadline=time.monotonic()+15
                hb=None
                while time.monotonic()<deadline:
                    msg=master.recv_match(type='HEARTBEAT',blocking=True,timeout=1)
                    if msg and msg.autopilot!=mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                        hb=msg
                        break
                if hb is None:
                    raise TimeoutError('No autopilot heartbeat')
                target=(hb.get_srcSystem(),hb.get_srcComponent())
                master.mav.command_long_send(*target,511,0,132,round(1e6/args.rate),0,0,0,0,0)
                clock=ClockMap()
                session=uuid.uuid4().hex
                counts=defaultdict(int)
                pending={}
                last_boot={}
                wall_offset=time.time_ns()-time.monotonic_ns()
                next_sync=next_print=0
                last_heartbeat=time.monotonic()
                sent=errors=0
                while True:
                    now=time.monotonic_ns()
                    offset=time.time_ns()-now
                    if abs(offset-wall_offset)>20_000_000:
                        session=uuid.uuid4().hex
                        clock=ClockMap()
                        counts.clear(); pending.clear(); last_boot.clear()
                        print('Pi wall clock stepped: new telemetry session; mapping reset')
                    wall_offset=offset
                    if now>=next_sync:
                        pending={k:v for k,v in pending.items() if now-v<2*NS}
                        # Exact nonce echo plus source filter supports MAVLink 1/2 safely.
                        pending[now]=now
                        master.mav.timesync_send(0,now)
                        next_sync=now+NS
                    msg=master.recv_match(blocking=True,timeout=.1)
                    received=time.monotonic_ns()
                    if time.monotonic()-last_heartbeat>5:
                        raise TimeoutError('Autopilot heartbeat lost')
                    if msg is None or (msg.get_srcSystem(),msg.get_srcComponent())!=target:
                        continue
                    kind=msg.get_type()
                    if kind=='HEARTBEAT':
                        last_heartbeat=time.monotonic()
                    elif kind=='TIMESYNC' and msg.tc1 and msg.ts1 in pending:
                        clock.add(pending.pop(msg.ts1),received,msg.tc1)
                    elif kind=='COMMAND_ACK' and msg.command==511:
                        print('Message interval ACK result:',msg.result)
                    elif kind=='DISTANCE_SENSOR':
                        key=(msg.id,msg.orientation)
                        boot=int(msg.time_boot_ms)
                        if key in last_boot and boot<last_boot[key]:
                            # Includes 49-day wrap: fail closed and re-establish mapping.
                            clock=ClockMap(); pending.clear(); counts.clear()
                            session=uuid.uuid4().hex
                            last_boot.clear()
                            print('Pixhawk clock reset/wrapped: new telemetry session')
                        last_boot[key]=boot
                        counts[key]+=1
                        p=make_packet(msg,session,counts[key],received)
                        p.update(schema=SCHEMA,pi_receive_unix_ns=time.time_ns(),
                                 sample_unix_ns=None,clock_uncertainty_ns=None,clock_status='UNMAPPED')
                        estimate=clock.estimate(received)
                        if estimate and boot>0:
                            off,uncertainty=estimate
                            sample_mono=boot*1_000_000+off
                            if -50_000_000<=received-sample_mono<=NS:
                                p.update(sample_unix_ns=sample_mono+(p['pi_receive_unix_ns']-received),
                                         clock_uncertainty_ns=uncertainty,clock_status='MAPPED_MESSAGE_TIME')
                        try:
                            udp.sendto(encode(p),destination)
                            sent+=1
                        except (BlockingIOError,OSError):
                            errors+=1
                        if received>=next_print:
                            print(f"{label(msg.orientation)} id={msg.id} range={p['distance_m']} m "
                                  f"clock={p['clock_status']} uncertainty_ms="
                                  f"{None if p['clock_uncertainty_ns'] is None else round(p['clock_uncertainty_ns']/1e6,2)} "
                                  f"sent={sent} send_errors={errors}")
                            next_print=received+NS
            except (OSError,TimeoutError) as exc:
                print('Serial reconnect:',exc)
                time.sleep(1)
            finally:
                if master is not None:
                    master.close()
                    master=None
    except KeyboardInterrupt:
        print('Stopped')
    finally:
        udp.close()

if __name__=='__main__':
    main()
