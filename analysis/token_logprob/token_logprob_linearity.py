import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator
import re

ROLLOUT_FILE = "evaluation/outputs/aime24_distill-qwen-1-5b.json"

CHECKPOINT_PATHS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "ckpt_path/global_step_100/actor/huggingface",
    # Add more checkpoint paths as needed
]

OUTPUT_DIR = "./outputs/distill-qwen-1-5b_grpo/token_logits_logprob/shards"

INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."


# ==================== Data Loading and Preparation ====================

def load_json(data_file):
    """Load JSON file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_responses(all_responses):
    """Flatten nested responses into individual items."""
    flattened_data = []
    for item in all_responses:
        raw_question = item["problem"]
        for generated_text in item["generated_texts"]:
            flattened_data.append({
                "raw_question": raw_question,
                "response_id": generated_text["response_id"],
                "response": generated_text["text"]
            })
    return flattened_data


def split_data(data, num_processes, process_index):
    """Split data across different processes."""
    total_samples = len(data)
    samples_per_process = (total_samples + num_processes - 1) // num_processes
    start_idx = process_index * samples_per_process
    end_idx = min(start_idx + samples_per_process, total_samples)
    return data[start_idx:end_idx]


# ==================== Model Setup ====================

def load_tokenizer(model_path):
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 128 * 1024
    return tokenizer


def load_model(model_path, accelerator):
    """Load the model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model = accelerator.prepare(model)
    return model


def compute_token_logprobs(logits, token_ids):
    """
    Compute log probability for each token.
    Args:
        logits: [seq_len, vocab_size] 
        token_ids: [seq_len]
    Returns:
        logprobs: [seq_len] log probability for each token
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs


# ==================== Inference Logic ====================

def prepare_inputs(raw_question, response, tokenizer):
    """Prepare model inputs."""
    prompt = raw_question + " " + INSTRUCTION_FOLLOWING
    message = [{"role": "user", "content": prompt}]
    raw_prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    full_text = raw_prompt + response
    
    # Tokenize
    prompt_tokens = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    
    prompt_length = prompt_tokens["input_ids"].shape[1]
    
    return inputs, prompt_length


def process_single_sample(data_item, model, tokenizer, device):
    """Process a single sample and return logits/logprobs."""
    raw_question = data_item["raw_question"]
    response = data_item["response"]
    
    # Prepare inputs
    inputs, prompt_length = prepare_inputs(raw_question, response, tokenizer)
    full_length = inputs["input_ids"].shape[1]
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Handle response part
    response_logits = logits[0, prompt_length-1:full_length-1, :]
    response_token_ids = inputs["input_ids"][0, prompt_length:]
    
    # Get token logits
    token_logits = response_logits.gather(
        dim=-1, 
        index=response_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    
    # Compute log probabilities
    response_logprobs = compute_token_logprobs(response_logits, response_token_ids)
    
    # Return results (move to CPU to save GPU memory)
    return {
        "token_ids": response_token_ids.cpu(),
        "token_logits": token_logits.cpu(),
        "logprobs": response_logprobs.cpu(),
    }


def process_data_batch(process_data, model, tokenizer, accelerator, step_name):
    """Process a batch of data."""
    process_results = []
    
    for data_item in tqdm(
        process_data, 
        desc=f"Rank {accelerator.process_index} - {step_name}",
        disable=not accelerator.is_local_main_process
    ):
        result = process_single_sample(data_item, model, tokenizer, accelerator.device)
        process_results.append(result)
    
    return process_results


# ==================== File Operations ====================

def get_step_name(model_path):
    """Extract step name from model path."""
    if "DeepSeek" in model_path or "Nemotron" in model_path:
        return "base_model"
    
    # Use regex to extract global_step number
    match = re.search(r'global_step_(\d+)', model_path)
    if match:
        return f"global_step_{match.group(1)}"
    
    # Fallback
    return "unknown_step"


def save_process_results(results, output_dir, step_name, process_index):
    """Save results for a single process."""
    process_output_file = os.path.join(
        output_dir, 
        f"{step_name}_logits_rank{process_index}.pt"
    )
    torch.save(results, process_output_file)
    return process_output_file


def merge_process_results(output_dir, step_name, num_processes):
    """Merge results from all processes."""
    all_results = []
    
    for rank in range(num_processes):
        rank_file = os.path.join(output_dir, f"{step_name}_logits_rank{rank}.pt")
        
        if os.path.exists(rank_file):
            print(f"Loading results from rank {rank}...")
            rank_results = torch.load(rank_file, map_location='cpu')
            all_results.extend(rank_results)
            
            # Remove temporary file
            os.remove(rank_file)
            print(f"Removed temporary file: {rank_file}")
        else:
            print(f"Warning: Expected file not found: {rank_file}")
    
    return all_results


def cleanup_memory(model, accelerator):
    """Clean up GPU memory."""
    del model
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    accelerator.free_memory()


# ==================== Main Workflow ====================

def process_checkpoint(model_path, process_data, tokenizer, accelerator):
    """Process a single checkpoint."""
    step_name = get_step_name(model_path)
    output_file = os.path.join(OUTPUT_DIR, f"{step_name}_logits.pt")
    
    # Check if output already exists
    if os.path.exists(output_file):
        if accelerator.is_main_process:
            print(f"Output file {output_file} already exists, skipping.")
        accelerator.wait_for_everyone()
        return
    
    if accelerator.is_main_process:
        print(f"\n{'='*50}")
        print(f"Loading model from {model_path}")
        print(f"Step: {step_name}")
        print(f"{'='*50}")
    
    # Load model
    model = load_model(model_path, accelerator)
    
    # Process data
    process_results = process_data_batch(process_data, model, tokenizer, accelerator, step_name)
    
    # Save process results
    if accelerator.is_local_main_process:
        print(f"Rank {accelerator.process_index}: Saving {len(process_results)} results")
    
    save_process_results(process_results, OUTPUT_DIR, step_name, accelerator.process_index)
    
    # Synchronize all processes
    accelerator.wait_for_everyone()
    
    # Main process merges results
    if accelerator.is_main_process:
        print(f"Merging results from all processes...")
        all_results = merge_process_results(OUTPUT_DIR, step_name, accelerator.num_processes)
        
        print(f"Saving merged results to {output_file}")
        torch.save(all_results, output_file)
        print(f"Successfully saved {len(all_results)} responses")
    
    # Clean up memory
    cleanup_memory(model, accelerator)
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print(f"Completed processing {step_name}\n")


def main():
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    accelerator.wait_for_everyone()
    
    # Load and prepare data
    all_responses = load_json(ROLLOUT_FILE)
    flattened_data = flatten_responses(all_responses)
    process_data = split_data(flattened_data, accelerator.num_processes, accelerator.process_index)
    
    if accelerator.is_main_process:
        print(f"Total samples: {len(flattened_data)}")
        print(f"Samples per process: {len(process_data)}")
        print(f"Number of processes: {accelerator.num_processes}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(CHECKPOINT_PATHS[0])
    
    # Process each checkpoint
    for model_path in CHECKPOINT_PATHS:
        if not os.path.exists(model_path):
            if accelerator.is_main_process:
                print(f"Warning: Path not found {model_path}, skipping.")
            continue
        
        process_checkpoint(model_path, process_data, tokenizer, accelerator)


if __name__ == "__main__":
    main()