# Results and limitations

## Model validation

| Variant | Best validation Dice in saved notebook output | Threshold displayed |
|---|---:|---:|
| B0 | 0.7711331026 | 0.50 |
| B1 | 0.7826975291 | 0.50 |
| B2 | 0.7909939782 | 0.50 |

Transcribed from the original explained-model notebook outputs before publication outputs were cleared. The source paths and SHA-256 hashes are in [the source manifest](source-manifest.json). The team reports training and test-data evaluation of all three variants. These particular numbers are checkpoint validation metadata, not held-out test measurements or a newly reproduced controlled comparison.

Older reports cite a different threshold (0.800) and test metrics. Those records are retained only in [the archive](archive/README.md). Their split/checkpoint/threshold relationship must be reconciled before presenting them as results of the current live package. Do not assume a missing checkpoint threshold is 0.5 or 0.8.

## Precision benchmark

The [Phase 1 summary](../04-live-implementation/examples/phase1-summary.json) records 100 sampled frames, three repetitions, and an RTX 3050 6 GB laptop GPU.

| Metric | FP32 | FP16 |
|---|---:|---:|
| Mean measured processing | 18.47 ms | 16.59 ms |
| Processing throughput | 54.15 FPS | 60.29 FPS |
| Peak allocated GPU memory | 224.57 MiB | 116.21 MiB |

The timed loop excludes RGB conversion/resizing already performed during sampling, capture, decoding, transport, display and saving. FP16 mask IoU against FP32 was 0.99824; that is precision-path agreement, not accuracy against ground truth. The recorded environment snapshot is historical evidence, not a guaranteed installable lockfile.

## Live sessions

| Measure | [Phase 3 session](../04-live-implementation/examples/phase3-summary.json) | [Phase 4 session](../04-live-implementation/examples/phase4-summary.json) |
|---|---:|---:|
| Duration | 54.05 s | 131.45 s |
| Frames received | 1,328 | 3,269 |
| Displayed or probed results | 909 | 2,228 |
| Mean processing | 24.75 ms | 23.75 ms |
| p95 processing | 35.69 ms | 29.97 ms |
| Both ranges matched | Not reported | 2,026 |

Phase 3 received 24.57 FPS and completed inference at 16.86 FPS. Phase 4 recorded 2,061 video-time-mapped results, 2,749 accepted telemetry packets and no reconnects or output drops in that session. Its gate is `MAPPING_OBSERVED_REQUIRES_HARDWARE_VALIDATION`; physical timing validation is false. Sessions differ in conditions and are not a controlled speed comparison. Their logs do not establish that they were airborne.

## Scope

- CNN/U-Net was not tested on real drone feed.
- Independent test manifests and duplicate checks are needed for reproducible model comparison.
- 256×256 inference may remove thin crack detail.
- Onboard segmentation, closed-loop crack following and validated physical crack sizing are not completed features of the live pipeline.
- ArUco/VIO/control files are separate experiments; some are empty scaffolding.
- Software tests cannot establish hardware timing, flight stability or full camera-to-screen delay.

## Visual demonstrations

The project [photos](project-gallery.md) and [flight GIFs](flight-demonstrations.md) show handheld camera tests, airborne operation and laptop segmentation overlays. They provide qualitative demonstration evidence. They do not identify which saved numerical session a photograph belongs to or supply reference masks for quantitative accuracy. The screen also displays approximate timing and, in some moments, unavailable sensor status; retain that distinction when interpreting the clips.
