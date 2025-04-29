#!/usr/bin/env bash

EXP_NAME=run_optimal_configs
declare -A ARCHITECTURES_REPRESENTATIONS=(
  [M5]="waveform,0.0001,0.01,cosine" # https://wandb.ai/dl-2-mm-jd/Hyperparameter%20sweep%20for%20ensemble%20submodels/runs/l5927ney/overview
  [Conformer]="mfcc,0.001,0.001,cosine" # https://wandb.ai/dl-2-mm-jd/Hyperparameter%20sweep%20for%20ensemble%20submodels/runs/loj86gwd/overview
  [ViT]="mfcc,0.001,0.001,cosine" # https://wandb.ai/dl-2-mm-jd/Hyperparameter%20sweep%20for%20ensemble%20submodels/runs/fw5cl7s6/overview
)

SEEDS=(216 123 999 476 8732)

for SEED in "${SEEDS[@]}"; do
  for ARCHITECTURE in "${!ARCHITECTURES_REPRESENTATIONS[@]}"; do
    IFS=',' read -r REPRESENTATION WEIGHT_DECAY LEARNING_RATE SCHEDULER <<< "${ARCHITECTURES_REPRESENTATIONS[$ARCHITECTURE]}"
    BASE_DIR="2_classes_${EXP_NAME}_${ARCHITECTURE}"

    python3 main.py \
      --config configuration/${EXP_NAME}/2_class.json \
      --model.architecture="$ARCHITECTURE" \
      --data.representation="$REPRESENTATION" \
      --training.weight_decay="$WEIGHT_DECAY" \
      --training.lr="$LEARNING_RATE" \
      --training.scheduler="$SCHEDULER" \
      --seed=$SEED \
      --project-dir="$BASE_DIR" \
      --wandb.name="2_class_${ARCHITECTURE}"
  done
done