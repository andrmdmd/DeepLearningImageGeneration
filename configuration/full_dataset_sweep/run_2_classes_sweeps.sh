#!/usr/bin/env bash

EXP_NAME=full_dataset_sweep
declare -A ARCHITECTURES_REPRESENTATIONS=(
  [M5]=waveform
  [Conformer]=mfcc
  [ViT]=mfcc
)

for ARCHITECTURE in "${!ARCHITECTURES_REPRESENTATIONS[@]}"; do
  REPRESENTATION=${ARCHITECTURES_REPRESENTATIONS[$ARCHITECTURE]}
  BASE_DIR="2_classes_${EXP_NAME}_${ARCHITECTURE}"

  python3 sweep.py \
    --config configuration/${EXP_NAME}/config_2_classes.json \
    --model.architecture="$ARCHITECTURE" \
    --data.representation="$REPRESENTATION" \
    --project-dir="$BASE_DIR" \
    --wandb.name="2_class_$ARCHITECTURE" \
    --sweep.name="2_class_$ARCHITECTURE"
done
