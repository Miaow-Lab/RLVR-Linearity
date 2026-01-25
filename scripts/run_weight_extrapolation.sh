#!/bin/bash

# ========== Configuration ==========
BASE_STEP=${1:-0}
SECOND_STEP=${2:-1400}
TARGET_STEP=${3:-2000}

# ========== Run ==========
python3 ./weight_extrapolation.py \
    --base_step ${BASE_STEP} \
    --second_step ${SECOND_STEP} \
    --target_step ${TARGET_STEP}