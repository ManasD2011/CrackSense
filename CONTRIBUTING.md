# Contributing

Keep training, Pi acquisition and laptop inference roles explicit. Preserve hardware experiments outside the default startup sequence. Update the phase README and file index whenever files move, and keep source provenance in `docs/source-manifest.json`.

Before submitting a change, run `python tools/check_repository.py` and `python -m unittest discover -s 04-live-implementation/tests -v`. The checker parses Python/notebook code and verifies current Markdown links without importing hardware scripts. The live suite needs NumPy, PyAV and OpenCV for its local integration test. Record skipped tests and do not report software checks as hardware or CUDA validation.

Do not commit dataset downloads, model weights, videos, virtual environments, Kaggle credentials, SSH keys or notebook outputs containing private material. Use placeholder addresses in new guides. Archived documents are preserved as historical evidence and are not current instructions.

A reuse license has not been selected; ask the project owner about redistribution permissions. Third-party datasets and dependencies retain their own terms.
