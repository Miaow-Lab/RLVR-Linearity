import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator
import argparse

# --- Static Path Configuration ---
PROMPT_PATH = "math-ai/aime24"
# Root output directory
OUTPUT_ROOT_DIR = "./outputs"
BASE_MODEL_0_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CKPT_ROOT_DIR = "ckpt_path"

DEFAULT_MAX_TOKENS = 32 * 1024
DEFAULT_TEMP = 0.6
DEFAULT_K = 8
DEFAULT_TOP_P = 0.95

# --- Helper Functions ---
def calculate_alpha(base_step, second_step, target_step):
    if second_step == base_step:
        raise ValueError("Second step cannot be equal to Base step.")
    return (target_step - base_step) / float(second_step - base_step)

def mix_logits(logits1, logits2, alpha, temperature=1.0):
    if alpha == 0.0:
        mixed_logits = logits1
    elif alpha == 1.0:
        mixed_logits = logits2
    else:
        mixed_logits = logits1 + alpha * (logits2 - logits1)
    
    if temperature > 0 and temperature != 1.0:
        mixed_logits = mixed_logits / temperature
    
    # Return Log Softmax
    return F.log_softmax(mixed_logits, dim=-1)

def top_p_filtering(logits, top_p=0.95, filter_value=-float('Inf')):
    """
    Apply Top-P (Nucleus) filtering to Logits.
    Note: The input 'logits' are actually log_softmax values (returned by mix_logits).
    We exp() them to get probabilities for cumulative sum calculation.
    """
    if top_p >= 1.0:
        return logits

    # 1. Sort (Descending)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(torch.exp(sorted_logits), dim=-1)

    # 3. Determine indices to remove (cumulative probability > top_p)
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # 4. Shift indices: Keep at least one token (even if the first token > top_p)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # 5. Restore original index order
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    
    # 6. Set logits of removed tokens to negative infinity
    logits[indices_to_remove] = filter_value
    return logits

def sample_from_logprobs(logp):
    return torch.distributions.Categorical(logits=logp).sample()

def generate_k_batched(
    model1, model2, tokenizer, prompt, k,
    alpha, max_new_tokens, temperature, top_p,
    device, tqdm_position: int = 0
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.expand(k, -1).to(device)

    with torch.no_grad():
        out1 = model1(input_ids=input_ids, use_cache=True, return_dict=True)
        logits1 = out1.logits[:, -1, :]
        past1 = out1.past_key_values
        out2 = model2(input_ids=input_ids, use_cache=True, return_dict=True)
        logits2 = out2.logits[:, -1, :]
        past2 = out2.past_key_values

    generated_ids = []
    is_finished = torch.zeros(k, dtype=torch.bool, device=device)
    active_indices = torch.arange(k, device=device)
    eos_token_ids = tokenizer.eos_token_id
    if not isinstance(eos_token_ids, (list, tuple)):
        eos_token_ids = [eos_token_ids]

    with tqdm(total=max_new_tokens, desc="  Gen Tokens", position=tqdm_position, leave=False, disable=False) as pbar:
        for i in range(max_new_tokens):
            if len(active_indices) == 0:
                pbar.update(max_new_tokens - pbar.n)
                break

            # 1. Mix Logits and apply Temperature
            logp_x = mix_logits(logits1, logits2, alpha, temperature)
            
            # 2. Apply Top-P filtering
            if top_p < 1.0:
                logp_x = top_p_filtering(logp_x, top_p=top_p)
            
            # 3. Sample
            next_ids = sample_from_logprobs(logp_x)
            
            step_generated_ids = torch.full((k,), tokenizer.pad_token_id, dtype=torch.long, device=device)
            step_generated_ids[active_indices] = next_ids
            generated_ids.append(step_generated_ids)

            is_finished[active_indices] = torch.isin(next_ids, torch.tensor(eos_token_ids, device=device))

            pbar.update(1)
            pbar.set_postfix({"Active": f"{len(active_indices)}/{k}"})

            # --- Dynamic shrinking ---
            keep_indices = (is_finished[active_indices] == False).nonzero(as_tuple=True)[0]
            if len(keep_indices) == 0:
                break

            active_indices = active_indices[keep_indices]
            logits1 = logits1[keep_indices]
            logits2 = logits2[keep_indices]
            
            if past1 is not None:
                temp_past1 = past1.reorder_cache(keep_indices)
                if temp_past1 is not None: past1 = temp_past1
            if past2 is not None:
                temp_past2 = past2.reorder_cache(keep_indices)
                if temp_past2 is not None: past2 = temp_past2
            
            next_input = next_ids[keep_indices].unsqueeze(-1)
            
            if hasattr(past1, "get_seq_length"):
                past_kv_length = past1.get_seq_length()
            else:
                past_kv_length = past1[0][0].shape[-2]

            pos_ids = torch.full((len(active_indices), 1), past_kv_length, dtype=torch.long, device=device)

            with torch.no_grad():
                out1 = model1(input_ids=next_input, position_ids=pos_ids, past_key_values=past1, use_cache=True, return_dict=True)
                out2 = model2(input_ids=next_input, position_ids=pos_ids, past_key_values=past2, use_cache=True, return_dict=True)
            
            logits1 = out1.logits[:, -1, :]
            logits2 = out2.logits[:, -1, :]
            past1 = out1.past_key_values
            past2 = out2.past_key_values

    if generated_ids:
        final_generated_ids = torch.stack(generated_ids, dim=1)
    else:
        final_generated_ids = torch.empty((k, 0), dtype=torch.long, device=device)

    responses = []
    for j in range(k):
        seq = final_generated_ids[j]
        pad_token_mask = (seq != tokenizer.pad_token_id)
        if pad_token_mask.any():
            first_non_pad = pad_token_mask.nonzero(as_tuple=True)[0][0]
            eos_indices = (seq[first_non_pad:] == eos_token_ids[0]).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                seq = seq[:first_non_pad + eos_indices[0]]
        responses.append(tokenizer.decode(seq, skip_special_tokens=True))
    return responses

def load_model(model_path, device):
    print(f"[Process {os.environ.get('LOCAL_RANK', 0)}] Loading model from {model_path} to {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device},
        trust_remote_code=True, attn_implementation="flash_attention_2"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prepare_prompts_hf(data_path, split='test'):
    ds = load_dataset(data_path)
    testset = ds[split]
    return list(testset)

def parse_args():
    parser = argparse.ArgumentParser(description="Logits Extrapolation (Decoupled & Organized)")
    parser.add_argument("--base_step", type=int, default=0)
    parser.add_argument("--second_step", type=int, default=500)
    parser.add_argument("--target_step", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P, help="Nucleus sampling probability threshold")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()
    
    # 1. Construct experiment-specific folder name (including top_p)
    exp_folder_name = f"exp_base{args.base_step}_2nd{args.second_step}_tgt{args.target_step}"
    exp_output_dir = os.path.join(OUTPUT_ROOT_DIR, exp_folder_name)
    
    # 2. Create directory (exist_ok=True for multi-process safety)
    os.makedirs(exp_output_dir, exist_ok=True)

    # Path construction
    if args.base_step == 0:
        model1_path = BASE_MODEL_0_PATH
    else:
        model1_path = os.path.join(CKPT_ROOT_DIR, f"global_step_{args.base_step}", "hf_model")
    model2_path = os.path.join(CKPT_ROOT_DIR, f"global_step_{args.second_step}", "hf_model")
    
    alpha = calculate_alpha(args.base_step, args.second_step, args.target_step)
    
    if accelerator.is_main_process:
        print(f"--- Configuration ---")
        print(f"Output Folder: {exp_output_dir}")
        print(f"Alpha: {alpha:.2f}")
        print(f"Temp: {args.temperature}, Top-P: {args.top_p}")
        print(f"---------------------")
    
    with accelerator.main_process_first():
        all_requests = prepare_prompts_hf(PROMPT_PATH)
        for idx, req in enumerate(all_requests):
            req["original_index"] = idx
    
    local_requests = all_requests[accelerator.process_index::accelerator.num_processes]
    device = accelerator.device
    
    try:
        model1, tokenizer = load_model(model1_path, device=device)
        model2, _ = load_model(model2_path, device=device)
    except OSError as e:
        print(f"[Error] Process {accelerator.process_index} failed to load models.")
        raise e
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    results = []
    
    outer_pos = accelerator.process_index * 2
    inner_pos = accelerator.process_index * 2 + 1
    
    # Outer progress bar: Display how many Requests processed
    for request in tqdm(local_requests, 
                        desc=f"GPU {accelerator.process_index} Reqs", 
                        disable=False, 
                        position=outer_pos):
        
        raw_question = request.get("problem", None)
        messages = [{"role": "user", "content": raw_question + ' ' + instruction_following}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        responses = generate_k_batched(
            model1=model1, model2=model2, tokenizer=tokenizer, prompt=prompt, k=args.k,
            alpha=alpha, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p,
            device=device, 
            tqdm_position=inner_pos
        )
        
        prompt_results = {
            "index": request["original_index"],
            "problem": raw_question,
            "gt": request.get("solution", None),
            "generated_texts": [{"response_id": j, "text": txt} for j, txt in enumerate(responses)]
        }
        results.append(prompt_results)
    
    del model1, model2
    torch.cuda.empty_cache()

    # Save to specific folder
    rank_filename = f"results_rank{accelerator.process_index}.json"
    rank_file_path = os.path.join(exp_output_dir, rank_filename)
    
    print(f"[Process {accelerator.process_index}] Saving to {rank_file_path}")
    with open(rank_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"[Process {accelerator.process_index}] Done.")

if __name__ == "__main__":
    main()