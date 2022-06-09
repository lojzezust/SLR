#!/bin/bash

# Arguments:
MASTR_DIR=data/mastr1325
ARCHITECTURE=wasr_resnet101_imu
MODEL_NAME=wasr_slr_v2
BATCH_SIZE=3
WARMUP_EPOCHS=25
FINETUNE_EPOCHS=50
NUM_ITER=2

# 1. Warm-up model
echo "1. Warm-up model"
python tools/train.py warmup \
    --architecture $ARCHITECTURE \
    --model-name ${MODEL_NAME}_it0 \
    --batch-size $BATCH_SIZE \
    --epochs $WARMUP_EPOCHS

for i in $(seq 1 $NUM_ITER)
do  
    let "PREV_I = $i - 1"
    CUR_I=$i
    PREV_VERSION=$(ls -t1 output/logs/${MODEL_NAME}_it${PREV_I} | head -n 1 | cut -d _ -f 2)

    PREV_MODEL_FILENAME="${MODEL_NAME}_it${PREV_I}_v${PREV_VERSION}"
    PREV_MODEL_WEIGHTS=output/logs/${MODEL_NAME}_it${PREV_I}/version_${PREV_VERSION}/checkpoints/last.ckpt
    FILLED_MASKS_DIR=output/pseudo_labels/${PREV_MODEL_FILENAME}

    echo "-------------------------"
    echo "Fine-tuning, iteration $CUR_I"
    echo "-------------------------"

    # 2. Estimate pseudo labels
    echo "2. Estimate pseudo labels"
    python tools/generate_pseudo_labels.py \
        --architecture $ARCHITECTURE \
        --weights-file $PREV_MODEL_WEIGHTS \
        --output-dir $FILLED_MASKS_DIR

    # 3. Re-train model
    echo "3. Retrain model"
    python tools/train.py finetune \
        --architecture $ARCHITECTURE \
        --model-name ${MODEL_NAME}_it${CUR_I} \
        --batch-size $BATCH_SIZE \
        --pretrained-weights $PREV_MODEL_WEIGHTS \
        --mask-dir $FILLED_MASKS_DIR \
        --epochs $FINETUNE_EPOCHS

done
