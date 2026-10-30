# Repository organization validation

Checked on 29 October 2026 during source organization.

- Python sources parse successfully and all published notebook code cells pass syntax checks after excluding IPython command lines.
- Notebook outputs and execution counts were cleared for publication. Historical model values were transcribed separately with source provenance.
- Current Markdown file links were checked. Explicitly archived documents retain historical references and are not startup instructions.
- All 14 live-pipeline software tests passed. The local H.264/UDP integration test decoded 62 frames, matched both sensors for 50 results, recovered one deliberate reconnect and verified recording. Counts can vary with scheduling.
- Pi sender and laptop dashboard command-line help work after folder separation.
- Known credential values and token/private-key patterns were checked without printing their contents. Credential files, model weights, videos, data downloads and environments are excluded.

## Changes needed for the new layout

Pi and laptop entrypoints resolve their sibling `shared/` directory by file location. Tests resolve the new device directories and launch the relocated dashboard. The Pi sender now requires `--host` explicitly instead of retaining an old network address. The sender test supplies a loopback destination. The integration test treats Windows' connection-aborted response like the existing reset/broken-pipe responses during deliberate reconnect/teardown.

A stray `q` after `return_dict=True` in the archived Phase 3 script was removed to repair its original syntax error. No inference algorithm was changed for this correction. The historical U-Net notebook explanation was corrected to identify pseudo-mask generation without asserting that every dataset release lacks pixel annotations.

## Not rerun

CNN/U-Net or SegFormer training, final held-out model evaluation, model/checkpoint CUDA inference, Raspberry Pi camera capture, Pixhawk communication, physical lidar timing, flight tests and camera-to-screen latency were not rerun. The local video integration test is software evidence, not a substitute for those measurements. Requirements are documented dependencies and historical environment records, not newly validated clean-install lockfiles.
