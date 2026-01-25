import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from pathlib import Path
import random
from tqdm import tqdm
from transformers import AutoTokenizer

# ==================== Configuration ====================

INPUT_DIR = "./outputs/distill-qwen-1-5b_grpo/token_logits_logprob/shards"
OUTPUT_DIR = "./outputs/distill-qwen-1-5b_grpo/token_logits_logprob/analysis"

TOKENIZER_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Visualization configuration
NUM_TOKENS_TO_SAMPLE = 100000  # Number of tokens to sample for analysis
RANDOM_SEED = 42            # Random seed for reproducibility

MIN_LOGPROB_CHANGE = 0.2

# ==================== DeepMind Style Coloring ====================
# Based on Google Research / DeepMind color palette (Colorblind-friendly)
DM_BLUE = "#0072B2"      # Primary: Dark Blue
DM_RED = "#D55E00"       # Emphasis: Vermilion (for fit lines or high change)
DM_CYAN = "#56B4E9"      # Secondary: Sky Blue (for low change)
DM_GREEN = "#009E73"     # Secondary: Bluish Green
DM_YELLOW = "#F0E442"    # Secondary: Yellow
DM_GREY = "#555555"      # Text/Generic Grey
DM_LIGHT_GREY = "#E0E0E0" # Grid lines

# Set global plotting style - Huge Fonts version
def set_deepmind_style():
    """Configure Matplotlib to use DeepMind style (Huge Fonts)."""
    plt.rcParams.update({
        # Font settings - significantly increase font size
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Roboto', 'Helvetica'],
        'font.size': 16,          # Base font 16 -> 26
        'axes.titlesize': 20,     # Subplot title 20 -> 34
        'axes.labelsize': 18,     # Axis labels 18 -> 32
        'xtick.labelsize': 15,    # X-axis ticks 15 -> 28
        'ytick.labelsize': 15,    # Y-axis ticks 15 -> 28
        
        # Colors and lines
        'text.color': '#212121',
        'axes.labelcolor': '#212121',
        'xtick.color': '#212121',
        'ytick.color': '#212121',
        
        # Axis style (remove borders)
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 2.5,    # Thicker lines
        'axes.edgecolor': '#424242',
        
        # Grid
        'axes.grid': True,
        'grid.color': '#EEEEEE',
        'grid.linestyle': '-',
        'grid.linewidth': 1.5,
        'axes.axisbelow': True, 
        
        # Legend
        'legend.frameon': False, 
        'legend.fontsize': 15,    # Legend font 15 -> 26
        
        # Background
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

# Mapping checkpoints to steps
CHECKPOINT_STEPS = {
    "base_model": 0,
    "global_step_100": 100,
    "global_step_200": 200,
    "global_step_300": 300,
    "global_step_400": 400,
    "global_step_500": 500,
    "global_step_600": 600,
    "global_step_700": 700,
    "global_step_800": 800,
    "global_step_900": 900,
    "global_step_1000": 1000,
    "global_step_1100": 1100,
    "global_step_1200": 1200,
    "global_step_1300": 1300,
}


# ==================== Tokenizer Loading ====================

def load_tokenizer(tokenizer_path):
    """Load tokenizer."""
    print(f"Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print("Tokenizer loaded successfully!")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Will proceed without token string conversion")
        return None


def token_id_to_str(token_id, tokenizer):
    """Convert token ID to string, handling special cases."""
    if tokenizer is None:
        return f"ID{token_id}"
    
    try:
        token_str = tokenizer.decode([token_id])
        # Handle special characters for display and filenames
        token_str = token_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        # Limit length
        if len(token_str) > 20:
            token_str = token_str[:17] + "..."
        # Handle empty strings or spaces
        if not token_str.strip():
            token_str = f"[SPACE_{token_id}]"
        return token_str
    except:
        return f"ID{token_id}"


def sanitize_filename(text, max_length=50):
    """Sanitize string to be suitable for filenames."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, '_')
    
    # Replace spaces
    text = text.replace(' ', '_')
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


# ==================== Data Loading ====================

def load_all_checkpoints(input_dir):
    """Load results from all checkpoints."""
    checkpoint_data = {}
    
    # Find all result files
    result_files = sorted(Path(input_dir).glob("*_logits.pt"))
    
    print(f"Found {len(result_files)} checkpoint files")
    
    for file_path in result_files:
        # Extract step name
        step_name = file_path.stem.replace("_logits", "")
        
        if step_name not in CHECKPOINT_STEPS:
            print(f"Warning: Unknown step name {step_name}, skipping")
            continue
        
        print(f"Loading {file_path.name}...")
        data = torch.load(file_path, map_location='cpu')
        
        checkpoint_data[step_name] = {
            'step': CHECKPOINT_STEPS[step_name],
            'data': data
        }
    
    # Sort by step
    checkpoint_data = dict(sorted(checkpoint_data.items(), key=lambda x: x[1]['step']))
    
    return checkpoint_data


# ==================== Data Organization ====================

def tensor_to_numpy(tensor):
    """Safely convert tensor to numpy array, handling BFloat16."""
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy()
    return tensor.numpy()


def organize_token_data_fast(checkpoint_data):
    """
    Optimized version: Use numpy arrays for batch processing, handling only logprobs.
    """
    step_names = list(checkpoint_data.keys())
    steps = np.array([checkpoint_data[name]['step'] for name in step_names])
    num_checkpoints = len(steps)
    
    # Get the first checkpoint to determine the number of responses
    first_checkpoint = checkpoint_data[step_names[0]]['data']
    num_responses = len(first_checkpoint)
    
    print(f"Processing {num_responses} responses across {num_checkpoints} checkpoints...")
    print(f"Steps: {steps}")
    
    response_data = []
    
    for response_idx in tqdm(range(num_responses), desc="Organizing responses"):
        # Get token_ids and length from the first checkpoint
        first_data = checkpoint_data[step_names[0]]['data'][response_idx]
        token_ids = tensor_to_numpy(first_data['token_ids'])
        num_tokens = len(token_ids)
        
        # Pre-allocate array: [num_tokens, num_checkpoints]
        logprobs_matrix = np.zeros((num_tokens, num_checkpoints), dtype=np.float32)
        
        # Batch fill data
        for ckpt_idx, step_name in enumerate(step_names):
            ckpt_data = checkpoint_data[step_name]['data'][response_idx]
            
            # Verify token count consistency
            if len(ckpt_data['token_ids']) != num_tokens:
                print(f"Warning: Response {response_idx} has inconsistent token count at {step_name}")
                continue
            
            # Convert to numpy, handle BFloat16
            logprobs_matrix[:, ckpt_idx] = tensor_to_numpy(ckpt_data['logprobs'])
        
        response_data.append({
            'response_idx': response_idx,
            'token_ids': token_ids,  # [num_tokens]
            'logprobs': logprobs_matrix,  # [num_tokens, num_checkpoints]
        })
    
    return {
        'steps': steps,
        'response_data': response_data
    }

# ==================== Token Sampling ====================

def sample_tokens(organized_data, num_samples, tokenizer, min_change=0.0, seed=42):
    """
    Randomly sample tokens for analysis, adding filtering conditions based on LogProb changes.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    response_data = organized_data['response_data']
    steps = organized_data['steps']
    
    print(f"Filtering tokens with absolute LogProb change > {min_change}...")
    
    # Collect eligible token indices
    eligible_token_indices = []
    total_tokens_scanned = 0
    
    for response in tqdm(response_data, desc="Filtering tokens"):
        response_idx = response['response_idx']
        # logprobs shape: [num_tokens, num_checkpoints]
        logprobs = response['logprobs']
        num_tokens = logprobs.shape[0]
        total_tokens_scanned += num_tokens
        
        # Vectorized calculation of change: last step - first step
        diffs = np.abs(logprobs[:, -1] - logprobs[:, 0])
        
        # Find indices meeting the criteria
        valid_indices = np.where(diffs > min_change)[0]
        
        for token_idx in valid_indices:
            eligible_token_indices.append({
                'response_idx': response_idx,
                'token_idx': int(token_idx), # Convert to Python int
                'abs_change': float(diffs[token_idx])
            })
    
    num_eligible = len(eligible_token_indices)
    print(f"Total tokens scanned: {total_tokens_scanned}")
    print(f"Tokens meeting criteria (> {min_change} change): {num_eligible} ({num_eligible/total_tokens_scanned*100:.2f}%)")
    
    if num_eligible == 0:
        print("Warning: No tokens met the criteria! Returning empty list.")
        return []
    
    # Random sample
    real_num_samples = min(num_samples, num_eligible)
    sampled_indices = random.sample(eligible_token_indices, real_num_samples)
    
    print(f"Sampled {real_num_samples} tokens from eligible set")
    
    # Extract sampled token data
    sampled_tokens = []
    for idx_info in tqdm(sampled_indices, desc="Extracting sampled tokens"):
        response_idx = idx_info['response_idx']
        token_idx = idx_info['token_idx']
        
        # Find corresponding response
        response = response_data[response_idx]
        token_id = int(response['token_ids'][token_idx])
        
        sampled_tokens.append({
            'response_idx': response_idx,
            'token_idx': token_idx,
            'token_id': token_id,
            'token_str': token_id_to_str(token_id, tokenizer),
            'steps': steps,
            'logprobs': response['logprobs'][token_idx],  # [num_checkpoints]
            'abs_change': idx_info['abs_change'] # Record change for later review
        })
    
    return sampled_tokens

# ==================== Vectorized Pearson Correlation Calculation ====================

def compute_pearson_r_squared_vectorized(x, Y):
    """
    Vectorized calculation of Pearson correlation coefficient squared.
    """
    n = len(x)
    m = Y.shape[0]
    
    # Compute means
    x_mean = np.mean(x)
    y_mean = np.mean(Y, axis=1, keepdims=True)  # [m, 1]
    
    # Compute slope and intercept
    x_diff = x - x_mean  # [n]
    y_diff = Y - y_mean  # [m, n]
    
    numerator = np.sum(y_diff * x_diff[np.newaxis, :], axis=1)  # [m]
    denominator = np.sum(x_diff ** 2)  # scalar
    
    slopes = numerator / denominator  # [m]
    intercepts = y_mean.squeeze() - slopes * x_mean  # [m]
    
    # Compute Pearson correlation coefficient
    x_std = np.std(x, ddof=1)  # Use sample standard deviation
    y_std = np.std(Y, axis=1, ddof=1)  # [m]
    
    covariance = np.sum(y_diff * x_diff[np.newaxis, :], axis=1) / (n - 1)  # [m]
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        pearson_r = covariance / (x_std * y_std)
        pearson_r = np.where(np.isfinite(pearson_r), pearson_r, 0.0)
    
    # R² = r²
    r_squared = pearson_r ** 2
    
    return slopes, intercepts, r_squared


def compute_linear_fits_for_sampled_tokens(sampled_tokens):
    """
    Compute linear fits for sampled tokens using Pearson r².
    """
    print(f"Computing linear fits for {len(sampled_tokens)} sampled tokens...")
    
    # Extract all data for vectorized calculation
    steps = sampled_tokens[0]['steps']  # All tokens share the same steps
    num_tokens = len(sampled_tokens)
    num_checkpoints = len(steps)
    
    # Build matrix
    logprobs_matrix = np.zeros((num_tokens, num_checkpoints), dtype=np.float32)
    
    for i, token_data in enumerate(sampled_tokens):
        logprobs_matrix[i] = token_data['logprobs']
    
    # Vectorized fit (using Pearson r²)
    print("Performing vectorized linear regression with Pearson r²...")
    logprob_slopes, logprob_intercepts, logprob_r2 = compute_pearson_r_squared_vectorized(steps, logprobs_matrix)
    
    # Add fit results to each token
    results = []
    for i, token_data in enumerate(sampled_tokens):
        token_data['logprob_fit'] = {
            'slope': float(logprob_slopes[i]),
            'intercept': float(logprob_intercepts[i]),
            'r_squared': float(logprob_r2[i]),
        }
        results.append(token_data)
    
    return results


# ==================== Statistical Analysis ====================

def plot_r_squared_distribution(sampled_tokens_with_fits, output_dir):
    """Plot R² distribution histogram - DeepMind Style."""
    print("\nPlotting R² distribution...")
    
    logprob_r2 = [t['logprob_fit']['r_squared'] for t in sampled_tokens_with_fits]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Use DeepMind Blue
    ax.hist(logprob_r2, bins=50, color=DM_BLUE, alpha=0.8, 
            edgecolor='white', linewidth=0.5, zorder=3)
               
    # Axis label font size
    ax.set_xlabel('$R^2$', fontsize=25)
    ax.set_ylabel('Count', fontsize=25)
    
    # Tick label font size
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5) 
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'r_squared_distribution.pdf')
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Saved R² distribution plot to {output_file}")
    plt.close()
    

def compute_statistics(sampled_tokens_with_fits, output_dir):
    """Compute and save statistics."""
    print("\nComputing statistics...")
    
    logprob_r2 = np.array([t['logprob_fit']['r_squared'] for t in sampled_tokens_with_fits])
    logprob_slopes = np.array([t['logprob_fit']['slope'] for t in sampled_tokens_with_fits])
    
    statistics = {
        'num_tokens_analyzed': len(sampled_tokens_with_fits),
        'r_squared_method': 'Pearson correlation coefficient squared',
        
        'logprob_r2': {
            'mean': float(np.mean(logprob_r2)),
            'std': float(np.std(logprob_r2)),
            'median': float(np.median(logprob_r2)),
            'min': float(np.min(logprob_r2)),
            'max': float(np.max(logprob_r2)),
            'q25': float(np.percentile(logprob_r2, 25)),
            'q75': float(np.percentile(logprob_r2, 75)),
        },
        
        'logprob_slope': {
            'mean': float(np.mean(logprob_slopes)),
            'std': float(np.std(logprob_slopes)),
            'median': float(np.median(logprob_slopes)),
            'min': float(np.min(logprob_slopes)),
            'max': float(np.max(logprob_slopes)),
        },
    }
    
    stats_file = os.path.join(output_dir, 'statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Saved statistics to {stats_file}")
    
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY (Pearson r²)")
    print("="*60)
    print(f"\nTokens Analyzed: {len(sampled_tokens_with_fits)}")
    print(f"\nLogProb R²:")
    print(f"  Mean:   {statistics['logprob_r2']['mean']:.4f} ± {statistics['logprob_r2']['std']:.4f}")
    print(f"  Median: {statistics['logprob_r2']['median']:.4f}")
    print(f"  Range:  [{statistics['logprob_r2']['min']:.4f}, {statistics['logprob_r2']['max']:.4f}]")
    
    print(f"\nLogProb Slope:")
    print(f"  Mean:   {statistics['logprob_slope']['mean']:.4e} ± {statistics['logprob_slope']['std']:.4e}")
    print(f"  Median: {statistics['logprob_slope']['median']:.4e}")
    print(f"  Range:  [{statistics['logprob_slope']['min']:.4e}, {statistics['logprob_slope']['max']:.4e}]")
    print("="*60 + "\n")

# ==================== Main Function ====================

def main():
    # Apply DeepMind style
    set_deepmind_style()
    
    print("="*60)
    print(f"Analysis Split Threshold (High/Low Change): {MIN_LOGPROB_CHANGE}")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 0. Load tokenizer
    print("\n[Step 0] Loading tokenizer...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    # 1. Load all checkpoint data
    print("\n[Step 1] Loading checkpoint data...")
    checkpoint_data = load_all_checkpoints(INPUT_DIR)
    print(f"Loaded {len(checkpoint_data)} checkpoints")
    
    # 2. Organize data
    print("\n[Step 2] Organizing data by token...")
    organized_data = organize_token_data_fast(checkpoint_data)
    print(f"Organized data for {len(organized_data['response_data'])} responses")
    
    # 3. Sample tokens
    print(f"\n[Step 3] Sampling {NUM_TOKENS_TO_SAMPLE} tokens...")
    sampled_tokens = sample_tokens(
        organized_data, 
        NUM_TOKENS_TO_SAMPLE,
        tokenizer,
        min_change=MIN_LOGPROB_CHANGE, # Pass threshold
        seed=RANDOM_SEED
    )
    
    if not sampled_tokens:
        print("No tokens sampled. Exiting.")
        return
    
    # 4. Linear fit (using Pearson r²)
    print("\n[Step 4] Computing linear fits with Pearson r²...")
    sampled_tokens_with_fits = compute_linear_fits_for_sampled_tokens(sampled_tokens)
    
    # 5. Compute statistics
    print("\n[Step 5] Computing statistics...")
    compute_statistics(sampled_tokens_with_fits, OUTPUT_DIR)
    
    # 6. Plot R² distribution
    print("\n[Step 6] Plotting R² distribution...")
    plot_r_squared_distribution(sampled_tokens_with_fits, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("Analysis completed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()