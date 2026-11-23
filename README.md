<div align="center">

# Lightweight PatchCore

### Toward Efficient Memory Bank-Based Anomaly Detection via Lightweight Backbone Adaptation

**Official research implementation**

**Junhyeok Im · Jeonghun Cho**

*IEEE 15th Global Conference on Consumer Electronics (GCCE), 2026*

![Conference](https://img.shields.io/badge/PAPER-IEEE%20GCCE%202026-2376B7?style=for-the-badge)
![Publication](https://img.shields.io/badge/IEEE%20XPLORE-PENDING-orange?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PYTORCH-IMPLEMENTATION-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**Presented at IEEE GCCE 2026. Publication in IEEE Xplore is pending.**

</div>

---

## Overview

This repository contains the implementation of **Lightweight PatchCore**, which replaces the original WideResNet50 backbone with MobileNetV3-Small for efficient industrial anomaly detection.

The approach retains PatchCore's memory bank and nearest-neighbor anomaly detection framework while substantially reducing computational cost and maintaining competitive anomaly detection and localization performance.

Pixel-level AUROC decreases by only **0.93 percentage points on MVTec AD** and **0.79 percentage points on VisA**, demonstrating a modest performance trade-off despite substantial efficiency gains.

## Method

![Lightweight PatchCore Pipeline](assets/pipeline.png)

- **Backbone:** MobileNetV3-Small (frozen, ImageNet-pretrained)
- **Feature layers:** `features.3` and `features.8`
- **Embedding dimension:** 128
- **Coreset sampling:** 10%
- **Nearest-neighbor search:** k = 1

## Results

Mean ± standard deviation over three random seeds.

| Dataset | Backbone | Image AUROC (%) | Pixel AUROC (%) |
|---|---|---:|---:|
| MVTec AD | WideResNet50 | **98.99 ± 0.08** | **98.12 ± 0.01** |
| MVTec AD | MobileNetV3-Small | 95.13 ± 0.25 | 97.19 ± 0.02 |
| VisA | WideResNet50 | **94.65 ± 0.07** | **98.52 ± 0.02** |
| VisA | MobileNetV3-Small | 92.12 ± 0.16 | 97.73 ± 0.03 |

## Qualitative Results

Visual comparison of anomaly localization results using WideResNet50 and MobileNetV3-Small.

<table>
  <tr>
    <th align="center">MVTec AD</th>
    <th align="center">VisA</th>
  </tr>
  <tr>
    <td width="50%">
      <img src="assets/qualitative_mvtec.png" width="100%" alt="MVTec AD Qualitative Results">
    </td>
    <td width="50%">
      <img src="assets/qualitative_visa.png" width="100%" alt="VisA Qualitative Results">
    </td>
  </tr>
</table>

## Computational Efficiency

Compared with the WideResNet50 baseline:

- **27.1× fewer** backbone parameters
- **176.2× fewer** backbone FLOPs
- **5.32× faster** prediction on MVTec AD
- **5.97× faster** prediction on VisA

Prediction speed was evaluated on an NVIDIA A40 GPU using the complete PatchCore prediction pipeline.

## Limitations

Although Lightweight PatchCore substantially reduces backbone complexity, this efficiency improvement comes with a modest reduction in anomaly detection accuracy, particularly at the image level.

Furthermore, the memory bank and nearest-neighbor search remain largely unchanged from the original PatchCore framework, potentially limiting further improvements in memory efficiency and end-to-end inference speed.

Finally, the experiments were conducted on MVTec AD and VisA, with inference speed evaluated on an NVIDIA A40 GPU. Performance on resource-constrained edge devices has not yet been validated.

## Future Work

Future work will focus on further improving the efficiency of Lightweight PatchCore through model quantization and memory bank optimization.

We also plan to investigate the trade-offs among numerical precision, anomaly detection performance, and computational efficiency, with the ultimate goal of enabling practical deployment on resource-constrained edge devices.

## Getting Started

### Installation

```bash
git clone https://github.com/augustinus2000/lightweight-patchcore.git
cd lightweight-patchcore
pip install -e .
```


### Datasets

- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://github.com/amazon-science/spot-diff)

### VisA Dataset Preparation

We use the official [VisA dataset](https://github.com/amazon-science/spot-diff) with the **1-class split** provided by Amazon Science.

The original dataset must be reorganized into an MVTec AD-like directory structure before running our experiments.

#### Step 1. Download and extract VisA

Run the following commands from the repository root:

```bash
mkdir -p data/visa/raw
cd data/visa

wget -c https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar

tar -xf VisA_20220922.tar -C raw

cd ../..
```

The extracted `raw` directory contains the original 12 VisA categories.

#### Step 2. Download the official preprocessing code

```bash
git clone https://github.com/amazon-science/spot-diff.git data/visa/official-spot-diff
```

We use the official `utils/prepare_data.py` script and `split_csv/1cls.csv` file to reproduce the one-class dataset split.

#### Step 3. Generate the 1-class dataset

Run the following command from the repository root:

```bash
python data/visa/official-spot-diff/utils/prepare_data.py --split-type 1cls --data-folder data/visa/raw --save-folder data/visa/visa_pytorch --split-file data/visa/official-spot-diff/split_csv/1cls.csv
```

The official script organizes normal and anomalous images into training and test subsets. It also converts anomalous ground-truth masks into binary masks, where 0 represents normal pixels and 255 represents anomalous pixels.

The resulting directory structure is:

```text
data/visa/visa_pytorch/1cls/
├── candle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   └── bad/
│   └── ground_truth/
│       └── bad/
├── capsules/
├── cashew/
└── ... (12 categories in total)
```

#### Step 4. Configure dataset paths

Set the dataset paths from the repository root:

```bash
export MVTEC_ROOT="$PWD/data/mvtec"
export VISA_ROOT="$PWD/data/visa/visa_pytorch/1cls"
```

The MVTec AD dataset should contain its 15 category directories directly under `MVTEC_ROOT`.

#### Image and Mask Transformations

Both MVTec AD and VisA use the following transformations during data loading:

1. Resize the shorter side to 256 pixels while preserving the aspect ratio.
2. Apply a center crop to obtain a 224 x 224 image.
3. Convert the images and masks into tensors.
4. Normalize RGB images using ImageNet mean and standard deviation.

Ground-truth masks undergo the same spatial transformations without ImageNet normalization.

These transformations are performed **at runtime** by the dataset loaders. The original image files are not permanently resized.

The implementation is provided in `src/patchcore/datasets/mvtec.py` and `src/patchcore/datasets/visa.py`.


### Run Experiments

```bash
# MobileNetV3-Small on MVTec AD
bash scripts/run_experiment.sh mvtec mnv3 0 0

# MobileNetV3-Small on VisA
bash scripts/run_experiment.sh visa mnv3 0 0
```

Use `wr50` instead of `mnv3` for the original backbone baseline. Repeat with seeds `0`, `1`, and `2` for three-seed evaluation.

## Citation

The paper was presented at IEEE GCCE 2026. The IEEE Xplore DOI will be added when available.

```bibtex
@inproceedings{im2026lightweightpatchcore,
  title     = {Toward Efficient Memory Bank-Based Anomaly Detection via Lightweight Backbone Adaptation},
  author    = {Im, Junhyeok and Cho, Jeonghun},
  booktitle = {2026 IEEE 15th Global Conference on Consumer Electronics (GCCE)},
  year      = {2026}
}
```

## Acknowledgments

This work is based on the original [PatchCore](https://github.com/amazon-science/patchcore-inspection) implementation (CVPR 2022).

Please refer to [LICENSE](LICENSE) and [NOTICE](NOTICE) for licensing and attribution. Dataset visualizations may have separate usage restrictions.
