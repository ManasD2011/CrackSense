# Timing and range association

Video and telemetry are independent streams. Latest arrival does not mean simultaneous acquisition.

The Pi sender maps Pixhawk `DISTANCE_SENSOR.time_boot_ms` through a rolling TIMESYNC estimate into Pi time. The laptop maps RTP timestamps through RTCP sender reports into the video source timeline. These are approximate message/presentation-time mappings, not verified physical measurement timestamps.

The receiver requires two consistent sender reports; mappings expire/reset on defined discontinuities. The clock estimator uses recent low-RTT observations with uncertainty checks. Per-sensor histories retain at most 300 packets, with at most 32 sensor identities. A frame receives the newest eligible sample at or before its mapped time. Default allowed sample age is 250 ms, including estimated timing uncertainty. A newer invalid sample suppresses an older valid reading. Unknown clocks do not fall back to latest-arrival matching.

Sender-session replacement, sequence checks and source-IP filtering address stale/reordered data; they are not cryptographic authentication. No deliberate waiting for future telemetry is introduced. Each result freezes the selected frame/sample association before inference.

`VIDEO_TIME_UNAVAILABLE` means the required video mapping is unavailable; it does not alone mean the lidar failed. `TELEMETRY_STALE`, `SENSOR_CLOCK_UNAVAILABLE`, `NO_PRIOR_SAMPLE` and clock-domain diagnostics distinguish other cases.

Inference latency excludes capture, network, decoding, annotation and physical display. Decode-to-display preparation also does not measure screen refresh. A successful mapping count is evidence of code-path operation; an independently timed physical event is needed to validate actual alignment.
