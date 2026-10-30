> Historical document: superseded by the [current repository guide](../../../README.md). Paths, status and claims below may be outdated.

# Dataset, Model, and Reported Results

## Dataset

The final project report records 34,596 crack image/mask pairs combined from CrackSeg9K and CrackVision12K. For the B2 run, the data was split into training, validation, and test subsets in an 80:10:10 ratio with a fixed seed. The report describes automatic image-mask pair discovery, mask binarization at a grayscale threshold of 127, and data augmentation applied only to training examples.

The report identifies a possible duplication of 11,298 CrackSeg9K pairs between a top-level directory and train/test folders. It says a hash-based duplicate check is pending. Treat the reported test result as subject to that caveat until the check is completed.

The dataset and training checkpoints are not included in this GitHub repository. The report cites public CrackSeg9K and CrackVision12K sources; review their current licenses and terms before redistributing data.

## Model

Crack detection is formulated as binary semantic segmentation. The selected model is SegFormer MiT-B2 with background and crack classes. The report describes training at 256 × 256 for 40 epochs using a weighted BCE, Dice, and Focal Tversky loss combination. The decision threshold was calibrated on validation data; the selected value was `0.800`.

## Reported held-out test metrics

| Metric | Value |
|---|---:|
| Test images | 3,461 |
| Threshold | 0.800 |
| Dice | 0.7968 |
| IoU | 0.6623 |
| Precision | 0.7823 |
| Recall | 0.8119 |

These are values reported in the final project Black Book. They have not been reproduced from this checkout because the training data, training code, checkpoint, and pinned environment are not included here. The report says the model was visually run on camera images, but that evaluation had no ground-truth masks and therefore no Dice or IoU score.

## Limitations to report alongside the metrics

- A possible 11,298-pair duplication in CrackSeg9K has not been checked by file hash.
- The B0, B1, and B2 comparison is not fully controlled; B1 quantitative results were incomplete.
- There is no quantitative ground-truth evaluation on images captured by the project drone.
- Downsampling to 256 × 256 can remove thin cracks or widen the restored mask.
- The calibrated threshold is not consistently stored/used by every script; the main detector uses per-frame Otsu thresholding.
- TTA scale mapping, severity estimation, and quantized packages are described as experimental and not fully evaluated.

## Earlier project figures

The Black Book mentions earlier U-Net Dice and CNN classification accuracy values but says their data splits, thresholds, and metric definitions are undocumented. Do not compare those numbers directly with the B2 pixel-level test metrics.
