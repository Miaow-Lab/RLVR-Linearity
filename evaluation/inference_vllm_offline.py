import json
import os
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import argparse

# ================= Configuration =================
OUTPUT_DIR = "./outputs"

DATA2PROMPT_PATH = {
    "aime24": "math-ai/aime24",
    "aime25": "math-ai/aime25",
    "math500": "HuggingFaceH4/MATH-500",
}


SPLIT = "test"

MAX_NEW_TOKENS = 32 * 1024
TEMPERATURE = 0.6
TOPP = 0.95
SEED = 42

N = 64
NUM_INSTANCES = 8  # Number of vLLM instances (total cards)
GPUS_PER_INSTANCE = 1  # Number of GPUs per instance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Path to HF model for vLLM.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distill-qwen-1-5b",
        help="Model name used for output filename.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="aime24",
        choices=list(DATA2PROMPT_PATH.keys()),
        help="Dataset name (used to select prompt path and output filename).",
    )
    parser.add_argument(
        "--problem_key",
        type=str,
        default="problem",
        help="Field name for the problem in the dataset.",
    )
    parser.add_argument(
        "--gt_key",
        type=str,
        default="solution",
        help="Field name for the ground-truth solution in the dataset.",
    )
    return parser.parse_args()


def prepare_prompts(data_path, split='test'):
    ds = load_dataset(data_path)
    print(f"Loading dataset from {data_path}...")
    testset = ds[split]
    return list(testset)


def split_list_evenly(lst, n):
    """
    Split list 'lst' into 'n' parts as evenly as possible.
    The remainder 'm' is distributed among the first 'm' chunks, ensuring the maximum length difference is not greater than 1.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def run_vllm_inference_on_gpus(
    requests_chunk,
    gpu_devices_list,
    worker_id,
    start_global_idx,
    model_path,
    problem_key,
    gt_key,
):
    # Set CUDA_VISIBLE_DEVICES environment variable
    gpu_devices = ",".join(map(str, gpu_devices_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    print(
        f"[Worker {worker_id}] Initializing vLLM on GPUs {gpu_devices} "
        f"processing {len(requests_chunk)} samples..."
    )

    llm = LLM(
        model=model_path,
        tensor_parallel_size=len(gpu_devices_list),
        dtype="auto",
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    all_raw_prompt_ids = []
    instruction_following = (
        "Let's think step by step and output the final answer within \\boxed{}."
    )

    for request in requests_chunk:
        raw_question = request.get(problem_key, "")
        question = raw_question + ' ' + instruction_following
        message = [
            {"role": "user", "content": question},
        ]
        raw_prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
        all_raw_prompt_ids.append(raw_prompt_ids)

    sampling_params = SamplingParams(
        n=N,
        temperature=TEMPERATURE,
        top_p=TOPP,
        max_tokens=MAX_NEW_TOKENS,
        skip_special_tokens=True,
        seed=SEED,
    )

    outputs = llm.generate(
        prompt_token_ids=all_raw_prompt_ids,
        sampling_params=sampling_params,
    )

    # Process results directly in the child process
    processed_results = []
    for i, output in enumerate(outputs):
        req = requests_chunk[i]
        prompt_result = {
            "index": start_global_idx + i,  # Keep global index
            "problem": req.get(problem_key, None),
            "gt": req.get(gt_key, None),
            "generated_texts": [],
        }

        for j, single_output in enumerate(output.outputs):
            prompt_result["generated_texts"].append(
                {
                    "response_id": j,
                    "text": single_output.text,
                }
            )
        processed_results.append(prompt_result)

    return processed_results


def save_results(all_results, model_name, data_name):
    all_results.sort(key=lambda x: x['index'])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{data_name}_{model_name}.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_results)} results to {output_path}...")


def main():
    args = parse_args()
    model_path = args.model_path
    model_name = args.model_name
    data_name = args.data
    problem_key = args.problem_key
    gt_key = args.gt_key

    prompt_path = DATA2PROMPT_PATH[data_name]

    print(f"Using MODEL_PATH: {model_path}")
    print(f"Using MODEL_NAME: {model_name}")
    print(f"Using DATA: {data_name}")
    print(f"Using PROMPT_PATH: {prompt_path}")
    print(f"Using PROBLEM_KEY: {problem_key}")
    print(f"Using GT_KEY: {gt_key}")

    # 1. Load data
    all_requests = prepare_prompts(prompt_path, SPLIT)

    total_samples = len(all_requests)
    print(f"Total samples: {total_samples}")

    # 2. Split data evenly
    instance_requests_list = split_list_evenly(all_requests, NUM_INSTANCES)

    # 3. Prepare GPU assignments
    gpu_assignments = []
    for i in range(NUM_INSTANCES):
        start_gpu = i * GPUS_PER_INSTANCE
        end_gpu = start_gpu + GPUS_PER_INSTANCE
        gpu_assignments.append(list(range(start_gpu, end_gpu)))

    # 4. Calculate start global index for each chunk for sorting during merge
    start_indices = []
    current_idx = 0
    for chunk in instance_requests_list:
        start_indices.append(current_idx)
        current_idx += len(chunk)

    # 5. Execute in parallel
    print(f"Starting inference with {NUM_INSTANCES} instances...")
    with ProcessPoolExecutor(max_workers=NUM_INSTANCES) as executor:
        futures = []
        for i in range(NUM_INSTANCES):
            if not instance_requests_list[i]:
                continue

            future = executor.submit(
                run_vllm_inference_on_gpus,
                instance_requests_list[i],
                gpu_assignments[i],
                i,
                start_indices[i],
                model_path,
                problem_key,
                gt_key,
            )
            futures.append(future)

        all_outputs = []
        for future in tqdm(futures, desc="Waiting for workers"):
            results = future.result()
            all_outputs.extend(results)

    # 6. Save results
    save_results(all_outputs, model_name, data_name)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()