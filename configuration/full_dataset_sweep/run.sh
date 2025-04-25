#!/usr/bin/env bash

for ((i=1; i<=3; i++))
do
    python3 sweep.py --config configuration/full_dataset_sweep/config.json --model.architecture=M5 --data.representation=waveform --project-dir=full_dataset_sweep_M5 --wandb.name=M5 --sweep.name=M5
    python3 sweep.py --config configuration/full_dataset_sweep/config.json --model.architecture=Conformer --data.representation=mfcc --project-dir=full_dataset_sweep_Conformer --wandb.name=Conformer --sweep.name=Conformer
    python3 sweep.py --config configuration/full_dataset_sweep/config.json --model.architecture=ViT --data.representation=mfcc --project-dir=full_dataset_sweep_ViT --wandb.name=ViT --sweep.name=ViT
done