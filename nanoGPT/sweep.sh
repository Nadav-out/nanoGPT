#!/bin/bash

SWEEP_ID=$1
NUM_GPUS=$2

for ((i=0;i<NUM_GPUS;i++)); do
    CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID &
done