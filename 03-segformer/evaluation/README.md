# Evaluation

**Execution:** training/evaluation computer. No separate final test runner or common B0/B1/B2 test manifest was found in the supplied source.

The training notebooks include `evaluate_full_metrics`, reporting pixel accuracy, precision, recall, F1, IoU and Dice on their training and validation loaders. That is not an independent test evaluation. The team performed test-data evaluation, but reproducible final split/configuration records for all three variants are not included here.

For a new comparison, record a deduplicated training, validation and test split. Fit the model on the training set, select the checkpoint and threshold on validation data, and evaluate once on the held-out test set. Use identical preprocessing and metric aggregation for all variants. Report checkpoint identity, threshold, resolution, counts and aggregation. Keep a separate drone-image test set with reference masks if evaluating airborne performance.

See [recorded results](../../docs/results-and-limitations.md). The walkthroughs show predictions and intermediate features, not a replacement for quantitative evaluation. FP16-versus-FP32 mask agreement also does not measure ground-truth accuracy.
