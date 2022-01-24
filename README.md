# SLR

A PyTorch implementation of the Scaffolded Learning Regime (SLR) from the WACV22 paper `Learning Maritime Obstacle Detection from Weak Annotations by Scaffolding`.

# Getting started

## Installation

1. Clone the repository
    ```bash
    git clone https://github.com/lojzezust/SLR
    cd SLR
    ```
2. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
3. Install SLR. Use the `-e` flag if you want to make changes.
    ```bash
    pip install -e .
    ```
4. Link datasets directory and create an output directory.
    ```bash
    ln -s path/to/data data
    mkdir output
    ```

## Preparing the data

1. Download the [MaSTr1325 dataset](https://box.vicos.si/borja/viamaro/index.html) and corresponding [weak annotations](https://github.com/lojzezust/SLR/releases/download/weights/mastr_extra.zip).
2. Use a script to prepare the data.
    ```bash
    python tools/prepare_data.py
    ```
    The preparation script performs the following operations:
    - Prepares object masks - converts bounding boxes from weak annotations into masks used in training
    - Prepares pairwise similarity maps - pre-computes the neighbor similarities used by the pairwise loss
    - Prepares partial masks - compute the partial masks used in the warm-up phase. Partial masks are constructed from weak annotations and IMU horizon masks.
    - Creates a datset file `all_weak.yaml`, which links the prepared dataset directories for training.

## SLR Training

### Step I: Feature warm-up

Train an initial model on partial labels generated from weak annotations and IMU. Uses additional object-wise losses.
```bash
export CUDA_VISIBLE_DEVICES=0,1
python tools/train.py warmup \
--architecture wasr_resnet101_imu \
--model-name wasr_slr_warmup \
--batch-size 4
```


### Step II: Generate pseudo labels

Generate pseudo labels by refining model predictions with learned features.
```bash
export CUDA_VISIBLE_DEVICES=0,1
python tools/generate_pseudo_labels.py \
--architecture wasr_resnet101_imu \
--weights-file output/logs/wasr_slr_warmup/version_0/checkpoints/last.ckpt \
--output-dir output/pseudo_labels/wasr_slr_warmup_v0
```

This creates the pseudo-labels and stores them into `output/pseudo_labels/wasr_slr_warmup_v0`.

### Step III: Fine-tune model

Fine-tune the initial model on the estimated pseudo-labels from the previous step. 
The model is initialized with weights of the initial model.

```bash
export CUDA_VISIBLE_DEVICES=0,1
python tools/train.py finetune \
--architecture wasr_resnet101_imu \
--model-name wasr_slr \
--batch-size 4 \
--pretrained-weights output/logs/wasr_slr_warmup/version_0/checkpoints/last.ckpt \
--mask-dir output/pseudo_labels/wasr_slr_warmup_v0
```
## Inference

### General inference

Run inference using a trained model. `tools/general_inference.py` script is able to run inference on a directory of images recursively. It replicates the directory structure in the output directory.

```bash
export CUDA_VISIBLE_DEVICES=0,1
python tools/general_inference.py \
--architecture wasr_resnet101 \
--weights-file output/logs/wasr_slr/version_0/checkpoints/last.ckpt \
--image-dir data/example_dir \
--output-dir output/predictions/test_predictions
```

Additionally, `--imu-dir` can be used to supply a directory with corresponding IMU horizon masks. The directory structure should match the one of image dir.

**NOTE**: The IMU dir has to be provided for models architectures relying on IMU data (i.e. WaSR).

### MODS inference

`tools/mods_inference.py` can be used in a similar fashion to run inference on the MODS benchmark.
# Pretrained models

Currently available pretrained model weights. All models are trained on the MaSTr1325 dataset using SLR and weak annotations.

| architecture       | backbone   | IMU | url                                                                                       |
|--------------------|------------|-----|-------------------------------------------------------------------------------------------|
| wasr_resnet101     | ResNet-101 |     | [weights](https://github.com/lojzezust/SLR/releases/download/weights/wasr_slr_rn101_noimu.pth)     |
| wasr_resnet101_imu | ResNet-101 |  ✓  | [weights](https://github.com/lojzezust/SLR/releases/download/weights/wasr_slr_rn101.pth) |

# Citation

Please cite this work as:
```bibtex
@inproceedings{Zust2022SLR,
  title={Learning Maritime Obstacle Detection from Weak Annotations by Scaffolding},
  author={{\v{Z}}ust, Lojze and Kristan, Matej},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={955--964},
  year={2022}
}
```
