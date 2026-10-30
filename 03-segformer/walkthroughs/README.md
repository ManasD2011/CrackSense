# Model walkthrough notebooks

**Execution:** laptop or training computer with the SegFormer environment and a compatible trained checkpoint. **Role:** inference explanation, not training.

- `CrackSense_Explained_B0.ipynb`: B0 checkpoint load, preprocessing, encoder features and mask thresholds.
- `CrackSense_Explained_B1.ipynb`: corresponding B1 walkthrough.
- `CrackSense_Explained_B2.ipynb`: B2 walkthrough used in deployment development.
- `CrackSense_Explained_B2_OriginalResolutionMasks.ipynb`: experimental original-resolution reconstruction and test-time augmentation.

Open Jupyter from this folder. Set `CHECKPOINT_PATH` and `IMAGE_PATH` to your files before running cells. The notebooks' default example images/checkpoint paths are historical placeholders and are not bundled. Use an absolute path or one relative to this folder. Select a checkpoint matching `MODEL_NAME`. Outputs have been cleared; run the notebook to regenerate them.

The normal explained path uses 256×256 and ImageNet normalization. Per-image Otsu comparisons are exploratory; live inference uses saved/explicit validated threshold metadata. Geometric/severity and TTA experiments are not the validated final live pipeline.
