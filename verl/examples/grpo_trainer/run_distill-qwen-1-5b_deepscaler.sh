#!/bin/bash
set -o pipefail
pip install fasttext
export CUDA_DEVICE_MAX_CONNECTIONS=1
PATH_ORI=${0%/*}
WORK_PATH=$(echo ${PATH_ORI} | sed -r 's/\/{2,}/\//')
echo "cur_path="${WORK_PATH}
cd ${WORK_PATH}


GPUS_PER_NODE=8
NODE_RANK=${RANK}
host="${HOSTNAME}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
NODE_NUM=${NODE_NUM} # WORLD_SIZE from env.
echo " - WORLD_SIZE:${NODE_NUM}"
TOTAL_GPUS=$(($GPUS_PER_NODE*$NODE_NUM))
distributed_options="--nproc_per_node $GPUS_PER_NODE --nnodes $NODE_NUM --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

export PYTHONPATH=${WORK_PATH}/../../:$PYTHONPATH
export RAY_PORT=6379
export RAY_ADDRESS="${MASTER_IP}:${RAY_PORT}"
export RAY_num_server_call_thread=1
export RAY_DEDUP_LOGS=0

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

if [ "${RANK}" = "0" ]; then
    ray start --head --port=${RAY_PORT} --include-dashboard=true --disable-usage-stats \
        2>&1 | tee -a ray_master_${RANK}.log
    sleep 30s
else
    ray start --address="${MASTER_ADDR}:${RAY_PORT}" --block \
        2>&1 | tee -a ray_worker_${RANK}.log
fi

MODEL_PATH=${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
CURRENT_STEP=${2:-0}
M_STEPS=${3:-100}
echo "Using MODEL_PATH: ${MODEL_PATH}"
echo "Current Step: ${CURRENT_STEP}"

train1_path=data_path/deepscaler-preview.parquet
train_files="['$train1_path']"
val1_path=data_path/aime24.parquet
val_files="['$val1_path']"
project_name="rl"
experiment_name="distill-qwen-1-5b_deepscaler_cur-step-${CURRENT_STEP}"
default_local_dir="./verl/outputs/ckpt/${project_name}/${experiment_name}"
echo "default_local_dir:${default_local_dir}"

# Create default_local_dir if it doesn't exist
if [ ! -d "${default_local_dir}" ]; then
    echo "Creating directory: ${default_local_dir}"
    mkdir -p "${default_local_dir}"
fi

if [ "${RANK}" = "0" ]; then
  RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --address ${RAY_ADDRESS} -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra', 'hf_model']" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=35840 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NODE_NUM} \
    trainer.default_local_dir=${default_local_dir} \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_training_steps=${M_STEPS} 2>&1 | tee -a ${WORK_PATH}/${experiment_name}_Rank${RANK}.log
fi