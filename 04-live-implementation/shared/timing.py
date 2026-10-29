"""Pure clock mapping and causal matching. All durations here are nanoseconds."""
from collections import deque
import json
import math
import statistics
import time
from telemetry_common import decode as decode_v1, encode

NS = 1_000_000_000
SCHEMA = 'cracksense.lidar.v2'

class ClockMap:
    """Rolling low-RTT TIMESYNC estimator; uncertainty is an estimate, not a bound."""
    def __init__(self):
        self.samples = deque(maxlen=20)

    def add(self, sent, received, remote):
        rtt = received - sent
        if not 0 < rtt <= 100_000_000 or remote <= 0:
            return False
        offset = (sent + received)//2 - remote
        if self.samples and abs(offset-self.samples[-1][2]) > 100_000_000:
            self.samples.clear()
        self.samples.append((received, rtt, offset))
        return True

    def estimate(self, now):
        recent = [x for x in self.samples if 0 <= now-x[0] <= 10*NS]
        if len(recent) < 3:
            return None
        best = min(recent, key=lambda x:x[1])
        spread = max(abs(x[2]-best[2]) for x in recent)
        uncertainty = best[1]//2 + spread + int((now-best[0])*0.0001) + 1_000_000
        if uncertainty > 25_000_000:
            return None
        return best[2], uncertainty


def decode(data):
    if len(data)>1400:
        raise ValueError('Oversized telemetry')
    p=json.loads(data)
    if not isinstance(p,dict) or p.get('schema')!=SCHEMA:
        raise ValueError('Use the new Phase 4 Pi sender (v2 schema)')
    base=dict(p, schema='cracksense.lidar.v1')
    decode_v1(encode(base))
    for k in ('pi_receive_unix_ns',):
        if type(p.get(k)) is not int or not 0 < p[k] < 2**63:
            raise ValueError(k)
    mapped=p.get('sample_unix_ns')
    uncertainty=p.get('clock_uncertainty_ns')
    if mapped is None:
        if uncertainty is not None or p.get('clock_status')!='UNMAPPED':
            raise ValueError('Invalid unmapped state')
    else:
        if type(mapped) is not int or not 0<mapped<2**63:
            raise ValueError('Invalid sample time')
        if type(uncertainty) is not int or not 0<=uncertainty<=25_000_000:
            raise ValueError('Invalid uncertainty')
        if not -50_000_000 <= p['pi_receive_unix_ns']-mapped <= NS:
            raise ValueError('Sample clock differs from Pi receive clock')
        if p.get('clock_status')!='MAPPED_MESSAGE_TIME':
            raise ValueError('Unknown clock status')
    return p


class History:
    """Caller owns locking. Histories are bounded and isolated by sender session."""
    def __init__(self):
        self.session=None
        self.retired=deque(maxlen=64)
        self.samples={}
        self.sequences={}
        self.accepted=self.rejected=self.gaps=0
        self.last_receive=None

    def add(self,p,received):
        session=p['session_id']
        if session in self.retired:
            self.rejected+=1
            return False
        if session!=self.session:
            if self.session is not None:
                self.retired.append(self.session)
            self.session=session
            self.samples.clear()
            self.sequences.clear()
        key=(p['system_id'],p['component_id'],p['sensor_id'],p['orientation'])
        previous=self.sequences.get(key,0)
        if p['sequence']<=previous or (key not in self.samples and len(self.samples)>=32):
            self.rejected+=1
            return False
        if previous:
            self.gaps+=max(0,p['sequence']-previous-1)
        self.sequences[key]=p['sequence']
        self.samples.setdefault(key,deque(maxlen=300)).append((dict(p),received))
        self.accepted+=1
        self.last_receive=received
        return True

    def match(self,frame_ns,video_uncertainty,orientation,sensor_id,now,max_age_ms=250):
        def missing(reason):
            return dict(status=reason,distance_m=None,sample_age_ms=None,clock_uncertainty_ms=None)
        if frame_ns is None:
            return missing('VIDEO_TIME_UNAVAILABLE')
        keys=[k for k in self.samples if k[2:]==(sensor_id,orientation)]
        if len(keys)!=1:
            return missing('SENSOR_MISSING' if not keys else 'AMBIGUOUS_SENSOR')
        history=self.samples[keys[0]]
        if now-history[-1][1]>NS:
            return missing('TELEMETRY_STALE')
        source_lag=history[-1][0]['pi_receive_unix_ns']-frame_ns
        if source_lag>NS:
            return missing('VIDEO_SOURCE_STALE')
        if source_lag < -NS:
            return missing('CLOCK_DOMAIN_MISMATCH')
        # An invalid newer sample must not be hidden behind an older valid value.
        if history[-1][0]['sample_unix_ns'] is None:
            return missing('SENSOR_CLOCK_UNAVAILABLE')
        candidates=[(p,r) for p,r in history if p['sample_unix_ns'] is not None and p['sample_unix_ns']<=frame_ns]
        if not candidates:
            return missing('NO_PRIOR_SAMPLE')
        p,_=max(candidates,key=lambda x:x[0]['sample_unix_ns'])
        age=frame_ns-p['sample_unix_ns']
        uncertainty=p['clock_uncertainty_ns']+(video_uncertainty or 0)
        out=dict(status=p['status'],distance_m=None,sample_age_ms=age/1e6,
                 clock_uncertainty_ms=uncertainty/1e6,sequence=p['sequence'],
                 sensor_id=p['sensor_id'],session_id=p['session_id'],
                 sample_unix_ns=p['sample_unix_ns'],timing='APPROXIMATE_MESSAGE_TIME')
        if uncertainty>50_000_000:
            out['status']='CLOCK_UNCERTAIN'
        elif age+uncertainty>max_age_ms*1e6:
            out['status']='SAMPLE_STALE'
        elif p['status']=='IN_RANGE':
            out['status']='MATCHED_APPROXIMATE'
            out['distance_m']=p['distance_m']
        return out
