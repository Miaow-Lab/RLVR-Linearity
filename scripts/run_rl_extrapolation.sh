#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# ========== Configuration ==========
# Training steps configuration
M_STEPS=100      # RL training steps per iteration
N_STEPS=100      # Weight extrapolation step increment
MAX_ITERATIONS=12 # Maximum number of iterations

# Paths
WEIGHT_EXTRAPOLATION_SCRIPT="./scripts/run_weight_extrapolation.sh"
RL_SCRIPT="./experiments/verl/examples/grpo_trainer/distill_qwen_1_5b_deepscaler.sh"

# Initial model path
INITIAL_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CURRENT_MODEL_PATH="${INITIAL_MODEL_PATH}"

# ========== Helper Functions ==========

# Clean up Ray cluster and release VRAM
cleanup_env() {
    echo "----------------------------------------"
    echo "🧹 Cleaning up Ray cluster and releasing VRAM..."
    ray stop --force || true
    sleep 5 # Wait for resources to be fully released
    echo "----------------------------------------"
}

# Trap exit or interrupt signals (Ctrl+C) to ensure environment cleanup
trap cleanup_env EXIT INT TERM

# Run RL training (Blocking mode)
run_rl_training() {
    local model_path=$1
    local current_step=$2
    local m_steps=$3
    
    echo "🚀 Running RL Training (Step ${current_step} -> $((current_step + m_steps)))"
    echo "   Model: ${model_path}"
    
    # Run directly; the script blocks until training finishes
    bash ${RL_SCRIPT} "${model_path}" "${current_step}" "${m_steps}"
    
    if [ $? -eq 0 ]; then
        echo "✅ RL Training script finished successfully."
        return 0
    else
        echo "❌ RL Training failed!"
        exit 1
    fi
}

# Run weight extrapolation
run_weight_extrapolation() {
    local base_step=$1
    local second_step=$2
    local target_step=$3
    
    echo "🧮 Running Weight Extrapolation: ${base_step} -> ${second_step} -> ${target_step}"

    bash ${WEIGHT_EXTRAPOLATION_SCRIPT} "${base_step}" "${second_step}" "${target_step}"
    
    if [ $? -eq 0 ]; then
        echo "✅ Weight extrapolation completed."
        return 0
    else
        echo "❌ Weight extrapolation failed."
        exit 1
    fi
}

# ========== Main Loop ==========

echo "========================================"
echo "Starting Alternating RL Training and Weight Extrapolation"
echo "M Steps (RL): ${M_STEPS}"
echo "N Steps (Extrapolation): ${N_STEPS}"
echo "Max Iterations: ${MAX_ITERATIONS}"
echo "========================================"

# Initialize step counters
current_base_step=0
iteration=0

while [ ${iteration} -lt ${MAX_ITERATIONS} ]; do
    iteration=$((iteration + 1))
    echo ""
    echo "######################################## "
    echo "### ITERATION ${iteration}/${MAX_ITERATIONS}"
    echo "### Start Time: $(date)"
    echo "########################################"
    echo ""
    
    # --- Step 1: Run RL training for M steps ---
    target_rl_step=$((current_base_step + M_STEPS))
    
    run_rl_training "${CURRENT_MODEL_PATH}" ${current_base_step} ${M_STEPS}
    
    # --- Step 2: Cleanup and Verify ---
    # After training exits, clean up Ray to release VRAM for the extrapolation script
    cleanup_env
    
    # Verify if the checkpoint was actually generated
    # Note: Ensure this path logic matches the actual output of your RL script
    checkpoint_dir="/verl/outputs/ckpt/rl/distill-qwen-1-5b_deepscaler_cur-step-${current_base_step}/global_step_${M_STEPS}"
    
    echo "🔍 Verifying checkpoint at: ${checkpoint_dir}"
    if [ -d "${checkpoint_dir}" ]; then
        echo "✅ Checkpoint found."
    else
        echo "❌ Critical Error: RL script finished but checkpoint directory not found!"
        echo "   Expected path: ${checkpoint_dir}"
        exit 1
    fi
    
    # --- Step 3: Run weight extrapolation ---
    target_extrapolation_step=$((target_rl_step + N_STEPS))
    
    # Determine base steps for extrapolation
    first_step=${current_base_step}
    second_step=${target_rl_step}
    
    run_weight_extrapolation ${first_step} ${second_step} ${target_extrapolation_step}
    
    # --- Step 4: Update model path for next iteration ---
    # Update model path (assuming extrapolation script saves to e{start}-{end}-{target})
    CURRENT_MODEL_PATH="outputs/e${first_step}-${second_step}-${target_extrapolation_step}"

    if [ ! -d "${CURRENT_MODEL_PATH}" ] && [ ! -f "${CURRENT_MODEL_PATH}/config.json" ]; then
         echo "⚠️ Warning: Extrapolated model path ${CURRENT_MODEL_PATH} does not seem to exist."
    fi

    current_base_step=${target_extrapolation_step}
    
    echo ""
    echo "🎉 Iteration ${iteration} completed."
    echo "   Next RL Start Step: ${current_base_step}"
    echo "   Next Model Path: ${CURRENT_MODEL_PATH}"
    echo ""
    
    sleep 3
done

echo "========================================"
echo "All iterations completed successfully!"
echo "Final step reached: ${current_base_step}"
echo "End Time: $(date)"
echo "========================================"