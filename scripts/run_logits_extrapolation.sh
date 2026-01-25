#!/bin/bash

# --- Core Experiment Parameters ---
BASE_STEPS=200
SECOND_STEP=1400
TARGET_STEP=2000

SCRIPT_PATH="./logits_extrapolation.py"

accelerate launch \
        --num_processes 8 \
        --num_machines 1 \
        --mixed_precision bf16 \
        --multi_gpu \
        "$SCRIPT_PATH" \
        --base_step $BASE_STEPS \
        --second_step $SECOND_STEP \
        --target_step $TARGET_STEP