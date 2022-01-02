
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
    The script performs the following operations.
    - Prepare object masks - converts bounding boxes from weak annotations into masks used in training
    - Prepare pairwise similarity maps - pre-computes the neighbor similarities used by the pairwise loss
    - Prepare partial masks - compute the partial masks used in the warm-up phase. Partial masks are constructed from weak annotations and IMU horizon masks.
    - Creates a datset file `all_weak.yaml`, which links the prepared dataset folders for training.

### Training

TODO