# SegFormer training notebooks

**Execution:** training computer, not Raspberry Pi.

`crack_detection_vit.ipynb` and `crack_detection_vit_1.ipynb` contain the actual training loop: dataset pairing, augmentation, model construction, loss, optimization, checkpointing and evaluation. They are historical development variants, not separately named final B0/B1/B2 runs. `_1` is an alternative revision, not a required second training step.

From the repository root after installing the phase environment:

```bash
cd 03-segformer/training
python -m jupyter lab
```

Choose one notebook. Run cells in order. The kernel working directory is this `training/` directory, so its `data/` and `checkpoints/` paths are local to that directory. Authenticate Kaggle through your personal configuration outside Git. Inspect the downloaded layout before accepting automatic image/mask matches.

## Configuration you must inspect

The configuration cell automatically chooses settings based on GPU memory. In the original notebook, some branches choose B0 and another chooses B2; the `_1` revision defaults to B0. To run a deliberate B0/B1/B2 experiment, explicitly set `MODEL_NAME` to `nvidia/mit-b0`, `nvidia/mit-b1` or `nvidia/mit-b2` **after the hardware-selection block**, and set image size, batch size, accumulation and epoch count. This is guidance for a new experiment, not a claim that those were the settings of every historical run.

The notebooks use RGB images, binary masks, augmentation and ImageNet normalization; cross-entropy plus Dice loss; separate encoder/head learning rates; mixed precision and gradient accumulation. They save optimizer/scheduler/scaler state for resuming. A checkpoint from a different architecture must not be loaded as the new variant.

The available notebooks shuffle paired samples and reserve about 10% for validation. They do not create a third independent test split. Do not call the validation loader a test set. See [evaluation](../evaluation/README.md) before reporting generalization results. Dataset pairing and duplicate checks are especially important for aggregated collections.

The final explained notebooks use 256×256 inputs, but these training revisions contain other input-size branches. Preserve and report the actual producing configuration. Complete final training logs/manifests for all three variants are not included. Notebook outputs are cleared for publication; selected historical metrics are transcribed in [results](../../docs/results-and-limitations.md).
