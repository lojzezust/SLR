
## Getting started

### Installation

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

### Preparing the data

1. Download the MaSTr1325 dataset and corresponding weak annotations
2. Use a script to prepare the data.
    ```bash
    python tools/prepare_data.py
    ```
    The preparation script performs the following operations:
    - Prepares object masks - converts bounding boxes from weak annotations into masks used in training
    - Prepares pairwise similarity maps - pre-computes the neighbor similarities used by the pairwise loss
    - Prepares partial masks - compute the partial masks used in the warm-up phase. Partial masks are constructed from weak annotations and IMU horizon masks.
    - Creates a datset file `all_weak.yaml`, which links the prepared dataset directories for training.

### Training

### Step I: Feature warm-up

Train an initial model on partial labels generated from weak annotations and IMU. Uses additional object-wise losses.
```bash
export CUDA_VISIBLE_DEVICES=0,1
python tools/train.py warmup \
--architecture wasr_resnet101_imu \
--model-name wasr_slr_warmup \
--train-file data/mastr1325/all_weak.yaml \
--val-file data/mastr1325/val.yaml \
--batch-size 4
```


### Step II: Generate pseudo labels

Generate pseudo labels by refining model predictions with learned features.
```bash
export CUDA_VISIBLE_DEVICES=0,1
python tools/generate_pseudo_labels.py \
--architecture wasr_resnet101_imu \
--weights_file output/logs/wasr_slr_warmup/version_0/checkpoints/last.ckpt \
--output_dir output/pseudo_labels/wasr_slr_warmup_v0
```
