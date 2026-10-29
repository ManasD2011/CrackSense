# Datasets

The project used:

- [CrackSeg9k on Kaggle](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset): project selection notes describe 9,255 images across 10 sub-datasets covering roads, walls, concrete, masonry and ceramic surfaces.
- [CrackVision12K on Kaggle](https://www.kaggle.com/datasets/vangiap/crackvision12k): project selection notes describe 12,000 images across 13 sub-datasets covering bridges, tunnels, concrete and pavement.

The headline total is 21,255, approximately 20,000 images. This is not an audited unique training-set count. Surface diversity and pixel-level mask supervision motivated the SegFormer work. Check the dataset terms at the sources before redistributing data.

The SegFormer training notebooks download and discover image/mask folders. They pair names and mask suffixes, convert images to RGB, and binarize grayscale masks. Inspect previews and verify correspondence: automatic folder discovery is not a dataset audit. The historical U-Net notebook instead generates pseudo-masks from class-folder images; do not confuse those targets with annotated segmentation masks.

Before a new reported run, save image/mask paths and hashes, remove duplicates, and keep related source images together when splitting. Create an independent test manifest in addition to training/validation. Preserve seeds and counts after filtering. Do not publish `kaggle.json` or API keys. No dataset files are bundled in this repository.
