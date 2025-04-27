#!/usr/bin/env bash

for ((i=1; i<=3; i++))
do
    # Replace the command_to_run with the actual command you want to run
    python3 sweep.py --config configuration/sampling_strategy/config.json
done