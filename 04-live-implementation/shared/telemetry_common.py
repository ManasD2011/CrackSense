"""Versioned lidar packet format shared by Pi sender and laptop receiver."""
import json
import math
import uuid

SCHEMA = "cracksense.lidar.v1"


def label(orientation):
    return {0: "FORWARD/WALL", 25: "DOWN/GROUND"}.get(orientation, f"ORIENT={orientation}")


def status(distance, minimum, maximum, quality):
    if distance <= 0 or distance >= 65532:
        return "INVALID"
    if maximum <= minimum:
        return "LIMITS_UNKNOWN"
    if not minimum <= distance <= maximum:
        return "OUT_OF_RANGE"
    if quality == 1:
        return "INVALID_QUALITY"
    return "IN_RANGE"


def make_packet(msg, session, sequence, received_ns):
    quality = int(getattr(msg, "signal_quality", 0))
    state = status(msg.current_distance, msg.min_distance, msg.max_distance, quality)
    return dict(
        schema=SCHEMA, session_id=session, sequence=sequence,
        system_id=msg.get_srcSystem(), component_id=msg.get_srcComponent(),
        sensor_id=int(msg.id), orientation=int(msg.orientation),
        raw_distance_cm=int(msg.current_distance), min_cm=int(msg.min_distance),
        max_cm=int(msg.max_distance), signal_quality=quality, status=state,
        distance_m=msg.current_distance / 100.0 if state == "IN_RANGE" else None,
        pixhawk_time_boot_ms=int(msg.time_boot_ms),
        pi_receive_monotonic_ns=received_ns,
    )


def encode(packet):
    return json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")


def decode(data):
    if len(data) > 1400:
        raise ValueError("oversized packet")
    p = json.loads(data)
    if not isinstance(p, dict) or p.get("schema") != SCHEMA:
        raise ValueError("unknown schema")
    if not isinstance(p.get("session_id"), str) or len(p["session_id"]) != 32:
        raise ValueError("invalid session ID")
    uuid.UUID(hex=p["session_id"])
    limits = {
        "sequence": (1, 2**63-1), "system_id": (1, 255), "component_id": (0, 255),
        "sensor_id": (0, 255), "orientation": (0, 255),
        "raw_distance_cm": (0, 65535), "min_cm": (0, 65535), "max_cm": (0, 65535),
        "signal_quality": (0, 100), "pixhawk_time_boot_ms": (0, 2**32-1),
        "pi_receive_monotonic_ns": (0, 2**63-1),
    }
    for field, (low, high) in limits.items():
        v = p.get(field)
        if type(v) is not int or not low <= v <= high:
            raise ValueError(f"invalid {field}")
    expected = status(p["raw_distance_cm"], p["min_cm"], p["max_cm"], p["signal_quality"])
    if p.get("status") != expected:
        raise ValueError("inconsistent validity")
    value = p.get("distance_m")
    if expected == "IN_RANGE":
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value-p["raw_distance_cm"]/100.0) > 1e-8:
            raise ValueError("inconsistent units")
    elif value is not None:
        raise ValueError("invalid measurement must have null distance")
    return p
