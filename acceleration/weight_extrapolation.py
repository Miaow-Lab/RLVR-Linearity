import torch
import gc
import os
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_PATHS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # Base model (step 0)
    "ckpt_path/global_step_100/hf_model",        # Second checkpoint
]

OUTPUT_DIR = "./outputs"

BASE_STEP = 0
SECOND_STEP = 1400
TARGET_STEP = 2000


def load_weights_only(model_path):
    """Load weights only to save memory, does not return the model object."""
    try:
        print(f"Loading weights from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        state_dict = model.state_dict()
        del model
        gc.collect()
        return state_dict
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None


def get_step_int_from_path(path):
    """Extract integer from 'global_step_XXX' in the path."""
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def main():
    base_model_path = CHECKPOINT_PATHS[0]
    second_model_path = CHECKPOINT_PATHS[1]

    # 1. Calculate Alpha
    # Formula: alpha = (Target - Base) / (Second - Base)
    if SECOND_STEP == BASE_STEP:
        raise ValueError("Second step equals Base step (division by zero). Cannot extrapolate.")

    alpha = (TARGET_STEP - BASE_STEP) / float(SECOND_STEP - BASE_STEP)

    # Dynamically generate output directory name
    save_folder_name = f"e{BASE_STEP}-{SECOND_STEP}-{TARGET_STEP}"
    save_path = os.path.join(OUTPUT_DIR, save_folder_name)

    print("=" * 50)
    print(f"Base Model Path:  {base_model_path}")
    print(f"Second Model Path:{second_model_path}")
    print("-" * 20)
    print(f"Base Step:        {BASE_STEP}")
    print(f"Second Step:      {SECOND_STEP}")
    print(f"Target Step:      {TARGET_STEP}")
    print("-" * 20)
    print(f"Formula:          ({TARGET_STEP} - {BASE_STEP}) / ({SECOND_STEP} - {BASE_STEP})")
    print(f"Calculated Alpha: {alpha:.4f}")
    print(f"Output Directory: {save_path}")
    print("=" * 50)

    # 2. Load Base Model (Container)
    print(f"Loading Base Model (Container) from {base_model_path}...")
    model_0 = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    trainable_keys = {name for name, param in model_0.named_parameters() if param.requires_grad}
    print(f"Identified {len(trainable_keys)} trainable parameters.")

    sd_0 = model_0.state_dict()

    # 3. Load Second Model (Weights only)
    sd_t = load_weights_only(second_model_path)
    if sd_t is None:
        return

    norm_keywords = ["layernorm", "rmsnorm", "ln_", "norm"]
    extrapolate_state_dict = {}

    # Lists to track parameter changes
    extrapolated_params = []   # Parameters that underwent linear extrapolation
    direct_copy_params = []    # Parameters copied directly from the second model

    print(f"Starting merge with alpha={alpha:.4f}...")
    with torch.no_grad():
        for key in sd_0.keys():
            if key not in sd_t:
                print(f"Warning: Key {key} not found in Model t, keeping original.")
                extrapolate_state_dict[key] = sd_0[key].to(torch.bfloat16)
                continue

            w0 = sd_0[key]
            wt = sd_t[key]

            is_norm_layer = any(keyword in key.lower() for keyword in norm_keywords)

            # Merge Logic
            if key in trainable_keys and not is_norm_layer:
                # --- Case A: Extrapolation ---
                # W_new = W_base + alpha * (W_second - W_base)
                delta = wt - w0
                w_new = w0 + alpha * delta
                extrapolate_state_dict[key] = w_new.to(torch.bfloat16)
                extrapolated_params.append(key)
            else:
                # --- Case B: Direct Copy from Second Model ---
                # For Norm layers or non-trainable parameters, it is safer to use the trained state directly
                extrapolate_state_dict[key] = wt.clone().to(torch.bfloat16)
                direct_copy_params.append(key)

    print("Merge complete. Updating model weights...")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 4. Load merged weights back into model_0
    print("Converting base model container to bfloat16...")
    model_0 = model_0.to(torch.bfloat16)
    model_0.load_state_dict(extrapolate_state_dict)

    print(f"Saving extrapolate model (bfloat16) to {save_path}...")
    model_0.save_pretrained(save_path)

    # 5. Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(second_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)

    # 6. Save parameter change log
    log_file_path = os.path.join(save_path, "merge_log.txt")
    print(f"Saving parameter log to {log_file_path}...")

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Merge Log\n")
        f.write(f"Base Step: {BASE_STEP}\n")
        f.write(f"Second Step: {SECOND_STEP}\n")
        f.write(f"Target Step: {TARGET_STEP}\n")
        f.write(f"Calculated Alpha: {alpha}\n")
        f.write(f"Base Model: {base_model_path}\n")
        f.write(f"Second Model: {second_model_path}\n")
        f.write("="*50 + "\n\n")

        f.write(f"### 1. Extrapolated Parameters (Count: {len(extrapolated_params)})\n")
        f.write(f"(Formula: Base + {alpha:.4f} * (Second - Base))\n")
        for k in sorted(extrapolated_params):
            f.write(f"{k}\n")
        f.write("\n")

        f.write(f"### 2. Direct Copy from Second (Count: {len(direct_copy_params)})\n")
        f.write("(Reason: Norm layer or Not Trainable)\n")
        for k in sorted(direct_copy_params):
            f.write(f"{k}\n")
        f.write("\n")

    print("Done!")


if __name__ == "__main__":
    main()