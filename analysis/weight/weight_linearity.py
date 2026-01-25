import torch
from transformers import AutoModelForCausalLM
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns 
import random
import os
import gc
import re
from tqdm import tqdm
from accelerate import Accelerator
import torch.distributed as dist
import matplotlib.ticker as ticker
import pandas as pd
import math

# ==========================================
# Configuration & Constants
# ==========================================

CHECKPOINT_PATHS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "ckpt_path/global_step_100/hf_model",
    # Add more checkpoint paths as needed
]

STEPS = torch.tensor([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300], dtype=torch.float32)

SAMPLE_PERCENTAGE = 0.001 
MIN_SAMPLES_THRESHOLD = 50 
MIN_UNIQUE_VALUES = 4
MIN_ABS_CHANGE = 1e-4

OUTPUT_DIR = "./outputs/distill-qwen-1-5b_grpo"
SEED = 42

# ==========================================
# DeepMind Style Configuration
# ==========================================

# DeepMind / Google Research Palette
DM_BLUE = "#0072B2"      # Primary: Dark Blue
DM_RED = "#D55E00"       # Emphasis: Vermilion
DM_GREEN = "#009E73"     # Secondary: Bluish Green
DM_YELLOW = "#F0E442"    # Secondary: Yellow
DM_CYAN = "#56B4E9"      # Secondary: Sky Blue
DM_GREY = "#555555"      # Text/Generic Grey
DM_LIGHT_GREY = "#E0E0E0" # Grid lines

def set_deepmind_style():
    """Configure Matplotlib to use DeepMind style."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Roboto', 'Helvetica'],
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        
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
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#424242',
        
        # Grid
        'axes.grid': True,
        'grid.color': '#EEEEEE',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'axes.axisbelow': True, # Grid behind data
        
        # Legend
        'legend.frameon': False, # No frame
        'legend.fontsize': 12,
        
        # Background
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

# ==========================================
# Utility Functions
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_weights_from_checkpoint(model_path):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="cpu", 
            low_cpu_mem_usage=True
        )
        state_dict = model.state_dict()
        del model 
        gc.collect()
        return state_dict
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def get_layer_weights(state_dict, layer_name, target_device):
    if layer_name in state_dict:
        return state_dict[layer_name].to(target_device)
    return None

def get_filtered_random_mask(layer_name, all_state_dicts, sample_ratio=0.01, min_samples=50, 
                             min_abs_change=1e-4, min_unique_values=4, device='cpu'):
    """
    Generate random mask and filter based on parameter changes over time.
    Only parameters meeting minimum change magnitude and minimum unique value counts are kept.
    """
    # 1. Get parameter shape (from first checkpoint)
    w0 = get_layer_weights(all_state_dicts[0], layer_name, device)
    if w0 is None: 
        return None
    
    shape = w0.shape
    num_params = shape.numel()
    del w0 # Free memory

    # 2. Generate initial candidate indices (random sampling)
    target_k = int(num_params * sample_ratio)
    k = max(min_samples, target_k)
    k = min(k, num_params)
    
    # Randomly select k indices
    candidate_indices = torch.randperm(num_params, device=device)[:k]
    
    # 3. Extract values for candidate parameters across all time steps
    history_list = []
    for sd in all_state_dicts:
        w = get_layer_weights(sd, layer_name, device)
        # Flatten and select only candidates
        w_flat = w.view(-1)
        selected_w = w_flat[candidate_indices]
        history_list.append(selected_w)
        del w
    
    # Y shape: [steps, k]
    Y = torch.stack(history_list).double()
    
    # 4. Apply filtering logic
    
    # 4.1 Filter parameters with small absolute changes (Min Abs Change)
    y_max = Y.max(dim=0).values
    y_min = Y.min(dim=0).values
    change_mask = (y_max - y_min) > min_abs_change
    
    # Get local indices passing the change test
    valid_indices_local = torch.where(change_mask)[0]
    
    if len(valid_indices_local) == 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)

    # 4.2 Filter parameters with too few unique values (Min Unique Values)
    final_valid_local_indices = []
    Y_filtered = Y[:, valid_indices_local]
    
    for i in range(Y_filtered.shape[1]):
        unique_vals = torch.unique(Y_filtered[:, i])
        if len(unique_vals) >= min_unique_values:
            original_local_idx = valid_indices_local[i]
            final_valid_local_indices.append(original_local_idx)
            
    if not final_valid_local_indices:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    
    final_valid_local_indices = torch.tensor(final_valid_local_indices, device=device)
    
    # 5. Map local indices back to global indices and generate mask
    final_global_indices = candidate_indices[final_valid_local_indices]
    
    mask_flat = torch.zeros(num_params, dtype=torch.bool, device=device)
    mask_flat[final_global_indices] = True
    
    return mask_flat.view(shape)

def shorten_layer_name(name):
    layer_num = re.search(r'layers\.(\d+)\.', name)
    l_str = f"L{layer_num.group(1)}" if layer_num else "Emb/Head"
    if "q_proj" in name: mod = "Q"
    elif "k_proj" in name: mod = "K"
    elif "v_proj" in name: mod = "V"
    elif "o_proj" in name: mod = "O"
    elif "gate_proj" in name: mod = "Gate"
    elif "up_proj" in name: mod = "Up"
    elif "down_proj" in name: mod = "Down"
    elif "embed" in name: mod = "Emb"
    elif "lm_head" in name: mod = "Head"
    else: mod = name.split('.')[-2]
    return f"{l_str}.{mod}"

# ==========================================
# Core Logic: Regression & Analysis
# ==========================================

def perform_vectorized_linear_regression(layer_name, mask, rl_steps, all_state_dicts, device):
    # 1. Extract data
    history_list = []
    for sd in all_state_dicts:
        w = get_layer_weights(sd, layer_name, device)
        selected_w = w[mask] 
        history_list.append(selected_w)
        del w 
    
    Y_final = torch.stack(history_list).double() 
    
    # If mask is empty or no data extracted
    if Y_final.shape[1] == 0:
        return np.array([]), np.array([]), np.array([])

    # 2. Linear regression
    steps_float = rl_steps.double().to(device)
    step_min = steps_float.min()
    step_max = steps_float.max()
    if step_max - step_min > 0:
        steps_norm = (steps_float - step_min) / (step_max - step_min)
    else:
        steps_norm = steps_float - step_min

    steps_gpu = steps_norm.view(-1, 1)
    num_steps = len(rl_steps)
    ones = torch.ones((num_steps, 1), device=device, dtype=torch.float64)
    A = torch.cat([ones, steps_gpu], dim=1) 

    try:
        result = torch.linalg.lstsq(A, Y_final)
        Beta = result.solution
    except Exception:
        A_pinv = torch.linalg.pinv(A)
        Beta = A_pinv @ Y_final

    Y_hat = A @ Beta
    
    # 3. Compute R2
    steps_mean = steps_norm.mean()
    steps_std = steps_norm.std()
    epsilon = 1e-10
    
    if steps_std <= epsilon:
        r2_scores = torch.zeros(Y_final.shape[1], device=device)
        return r2_scores.float().cpu().numpy(), Y_final.float().cpu().numpy(), Y_hat.float().cpu().numpy()
    
    steps_centered = steps_norm - steps_mean
    y_mean = Y_final.mean(dim=0, keepdim=True)
    y_std = Y_final.std(dim=0, keepdim=True)
    y_std = torch.where(y_std > epsilon, y_std, torch.ones_like(y_std))
    y_centered = Y_final - y_mean
    
    numerator = (steps_centered.view(-1, 1) * y_centered).sum(dim=0)
    denominator = (num_steps - 1) * steps_std * y_std.squeeze()
    pearson_r = numerator / (denominator + epsilon)
    r2_scores = torch.clamp(pearson_r ** 2, 0.0, 1.0)
        
    return (
        r2_scores.float().cpu().numpy(), 
        Y_final.float().cpu().numpy(),       
        Y_hat.float().cpu().numpy()    
    )

# ==========================================
# Plotting Functions
# ==========================================

def parse_layer_info(layer_name):
    """Parse layer name, return (Layer_Index, Module_Type)."""
    layer_match = re.search(r'layers\.(\d+)\.', layer_name)
    
    if layer_match:
        layer_idx = int(layer_match.group(1))
        if "input_layernorm" in layer_name: mod_type = "Norm_Attn_In"
        elif "post_attention_layernorm" in layer_name: mod_type = "Norm_MLP_In"
        elif "q_proj" in layer_name: mod_type = "Attn_Q"
        elif "k_proj" in layer_name: mod_type = "Attn_K"
        elif "v_proj" in layer_name: mod_type = "Attn_V"
        elif "o_proj" in layer_name: mod_type = "Attn_Out"
        elif "gate_proj" in layer_name: mod_type = "MLP_Gate"
        elif "up_proj" in layer_name: mod_type = "MLP_Up"
        elif "down_proj" in layer_name: mod_type = "MLP_Down"
        else:
            parts = layer_name.split('.')
            mod_type = parts[-2] if len(parts) > 1 else "Other"
        return layer_idx, mod_type
    else:
        if "embed_tokens" in layer_name: return -1, "Embedding"
        if "lm_head" in layer_name: return 29, "LM_Head"
        if "norm" in layer_name and "layers" not in layer_name: return 28, "Final_Norm"
        return None, None

def plot_heatmap_stats(layer_r2_means, layer_names):
    """
    Plot R2 distribution (Heatmap).
    """
    if not layer_names: return

    data = []
    for name, r2 in zip(layer_names, layer_r2_means):
        l_idx, m_type = parse_layer_info(name)
        if l_idx is not None:
            data.append({"Layer_Index": l_idx, "Module_Type": m_type, "R2": r2})
    
    if not data: return
    df = pd.DataFrame(data)

    special_indices = [-1, 28, 29]
    df_special = df[df['Layer_Index'].isin(special_indices)].copy()
    df_blocks = df[~df['Layer_Index'].isin(special_indices)].copy()

    # Use Nature style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    sns.set_context("paper", font_scale=1.4) 
    sns.set_style("white")

    cmap = "Blues" 
    vmin, vmax = 0.0, 1.0
    
    fig = plt.figure(figsize=(23, 10)) 
    gs = matplotlib.gridspec.GridSpec(1, 4, width_ratios=[35, 1.2, 0.4, 0.8], wspace=0.05)
    
    ax_main = fig.add_subplot(gs[0])
    ax_special = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[3]) 

    if not df_blocks.empty:
        heatmap_data = df_blocks.pivot(index="Module_Type", columns="Layer_Index", values="R2")
        block_order = ["Norm_Attn_In", "Attn_Q", "Attn_K", "Attn_V", "Attn_Out", 
                       "Norm_MLP_In", "MLP_Gate", "MLP_Up", "MLP_Down"]
        
        existing_rows = [r for r in block_order if r in heatmap_data.index]
        remaining_rows = [r for r in heatmap_data.index if r not in block_order]
        heatmap_data = heatmap_data.reindex(existing_rows + remaining_rows)
        heatmap_data.sort_index(axis=1, inplace=True)

        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, 
                    cbar=False, linewidths=0.5, linecolor='black',
                    clip_on=False, annot_kws={"size": 9}, vmin=vmin, vmax=vmax, ax=ax_main)
        
        ax_main.set_title("Transformer Blocks", fontsize=16, fontweight='bold', pad=15)
        ax_main.set_xlabel("Layer Depth", fontsize=14, fontweight='bold')
        ax_main.set_ylabel("Module Type", fontsize=14, fontweight='bold', labelpad=20)
        ax_main.set_yticklabels(ax_main.get_yticklabels(), rotation=90, va='center', fontsize=11)
        ax_main.tick_params(axis='y', pad=8)

    if not df_special.empty:
        name_map = {"Embedding": "Embedding", "Final_Norm": "Final Norm", "LM_Head": "LM Head"}
        df_special['Display_Name'] = df_special['Module_Type'].map(name_map)
        sorter = ["Embedding", "Final_Norm", "LM_Head"]
        df_special['Module_Type'] = pd.Categorical(df_special['Module_Type'], categories=sorter, ordered=True)
        df_special.sort_values('Module_Type', inplace=True)
        special_heatmap_data = df_special.set_index('Display_Name')[['R2']]

        sns.heatmap(special_heatmap_data, annot=True, fmt=".2f", cmap=cmap,
                    cbar=True, cbar_ax=ax_cbar, cbar_kws={'label': 'Linearity ($R^2$)'},
                    linewidths=0.5, linecolor='black', annot_kws={"size": 9},
                    vmin=vmin, vmax=vmax, ax=ax_special)
        
        ax_special.set_title("Special", fontsize=16, fontweight='bold', pad=15)
        ax_special.set_ylabel("") 
        ax_special.set_xlabel("")
        ax_special.yaxis.tick_right()
        ax_special.set_yticklabels(ax_special.get_yticklabels(), rotation=270, va='center', fontsize=12)
        ax_special.set_xticks([])

    ax_cbar.tick_params(labelsize=10)
    ax_cbar.set_ylabel('Linearity ($R^2$)', fontsize=12, fontweight='bold', labelpad=10)
    for spine in ax_cbar.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.suptitle(f"Linearity Analysis: DeepSeek-R1-Distill-Qwen-1.5B", fontsize=18, fontweight='bold', y=0.96)

    save_path = os.path.join(OUTPUT_DIR, "heatmap_nature_final_spaced.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved final spaced heatmap to {save_path}")


def plot_layer_fit(layer_name, rl_steps, r2_scores, actual_history, fitted_history, n_plots=4):
    """
    Plot layer fit - DeepMind Style
    """
    # Apply DeepMind style
    set_deepmind_style()
    
    num_params = actual_history.shape[1]
    if num_params == 0: return
    n_plots = min(n_plots, num_params)
    
    plot_indices = random.sample(range(num_params), n_plots)
    
    # Calculate grid
    cols = int(math.ceil(math.sqrt(n_plots)))
    rows = int(math.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    rl_steps_np = rl_steps.numpy()

    for i, idx in enumerate(plot_indices):
        ax = axes[i]
        if i < n_plots:
            # DeepMind Style Plotting
            # Actual points: DM_BLUE
            ax.scatter(rl_steps_np, actual_history[:, idx], color=DM_BLUE, alpha=0.6, s=80, 
                       label='Actual', edgecolors='none', zorder=2)
            # Fit line: DM_RED
            ax.plot(rl_steps_np, fitted_history[:, idx], color=DM_RED, linestyle='--', linewidth=3, 
                    label='Linear Fit', zorder=3)
            
            ax.set_title(f"Param #{idx} | $R^2$={r2_scores[idx]:.3f}", fontsize=22, pad=20, fontweight='medium')
            ax.legend(loc='best')
            
            # Format Y-axis with scientific notation
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 4)) 
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_fontsize(15)
            
            # Only show labels for bottom and left plots
            if i // cols == rows - 1:
                ax.set_xlabel('Step', fontsize=22)
            if i % cols == 0:
                ax.set_ylabel('Weight', fontsize=22)

            # Set tick label size
            ax.tick_params(axis='both', which='major', labelsize=18)
        else:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 1.0])
    save_path = os.path.join(OUTPUT_DIR, "layer_plots")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{layer_name.replace('.', '_')}_fit.pdf"), format='pdf', bbox_inches='tight')
    plt.close()


# ==========================================
# Main Processing Logic
# ==========================================

def load_all_checkpoints(accelerator):
    if accelerator.is_main_process:
        print(f"Loading checkpoints into CPU RAM...")
    
    all_state_dicts = []
    for p in CHECKPOINT_PATHS:
        sd = load_weights_from_checkpoint(p)
        if sd is not None:
            all_state_dicts.append(sd)
        else:
            if accelerator.is_main_process:
                print(f"Skipping {p}")
    return all_state_dicts

def process_single_layer(layer_name, all_state_dicts, device):
    """Logic for a single layer: sampling, regression, plotting."""
    
    # Default configuration
    current_sample_ratio = SAMPLE_PERCENTAGE
    current_min_abs_change = MIN_ABS_CHANGE
    current_min_unique = MIN_UNIQUE_VALUES

    # Special handling for Norm/Embedding layers
    # if "norm" in layer_name.lower():
    #     current_sample_ratio = 1.0  # Full sampling for Norm layers
    #     current_min_abs_change = -1e-9  # Allow tiny changes
    #     current_min_unique = 1         
    
    # Call mask generation function with filtering
    mask = get_filtered_random_mask(
        layer_name=layer_name,
        all_state_dicts=all_state_dicts,
        sample_ratio=current_sample_ratio,
        min_samples=MIN_SAMPLES_THRESHOLD,
        min_abs_change=current_min_abs_change, 
        min_unique_values=current_min_unique,  
        device=device
    )
    
    if mask is None or mask.sum() == 0: 
        return None

    # Perform regression
    r2, Y_actual, Y_fit = perform_vectorized_linear_regression(
        layer_name, mask, STEPS, all_state_dicts, device
    )
    
    if len(r2) == 0: return None
    
    # Plot fit for this layer (DeepMind Style)
    # Note: Multi-process plotting might have issues, ensuring unique filenames
    plot_layer_fit(layer_name, STEPS, r2, Y_actual, Y_fit)
    
    return {
        "name": layer_name,
        "mean_r2": np.mean(r2),
        "all_r2": r2 
    }

def run_distributed_analysis(accelerator, all_state_dicts):
    """Distribute layers and run analysis."""
    device = accelerator.device
    W0_dict = all_state_dicts[0]
    
    all_layer_names = [k for k in W0_dict.keys() if k.endswith(".weight")]
    all_layer_names.sort()

    # Task distribution
    my_layer_names = all_layer_names[accelerator.process_index::accelerator.num_processes]

    if accelerator.is_main_process:
        print(f"Total layers: {len(all_layer_names)}. Analyzing distributedly...")

    local_results = []
    for layer_name in tqdm(my_layer_names, desc=f"Process {accelerator.process_index}", position=accelerator.process_index):
        res = process_single_layer(layer_name, all_state_dicts, device)
        if res:
            local_results.append(res)
        torch.cuda.empty_cache()
        
    return local_results


def save_statistics(final_results):
    """Compute and save statistics (CSV and TXT)."""
    print("Calculating and saving statistics...")
    
    # 1. Prepare per-layer statistics
    layer_stats_data = []
    all_r2_values_global = []

    for res in final_results:
        l_name = res['name']
        l_r2s = res['all_r2']
        all_r2_values_global.append(l_r2s)
        
        # Filter invalid values
        l_r2s_clean = l_r2s[np.isfinite(l_r2s)]
        
        if len(l_r2s_clean) > 0:
            layer_stats_data.append({
                "Layer": l_name,
                "Mean_R2": np.mean(l_r2s_clean),
                "Median_R2": np.median(l_r2s_clean),
                "Std_R2": np.std(l_r2s_clean),
                "Sample_Count": len(l_r2s_clean)
            })
    
    # Save layer statistics to CSV
    if layer_stats_data:
        df_layers = pd.DataFrame(layer_stats_data)
        df_layers.sort_values(by="Layer", inplace=True)
        csv_path = os.path.join(OUTPUT_DIR, "layer_r2_stats.csv")
        df_layers.to_csv(csv_path, index=False)
        print(f"Saved layer statistics to {csv_path}")

    # 2. Compute global statistics
    if all_r2_values_global:
        combined_r2 = np.concatenate(all_r2_values_global)
        combined_r2 = combined_r2[np.isfinite(combined_r2)]
        valid_indices = (combined_r2 >= -1.0) & (combined_r2 <= 1.0) 
        filtered_r2 = combined_r2[valid_indices]
        
        if len(filtered_r2) > 0:
            # Plot histogram - DeepMind Style
            set_deepmind_style()
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Use DeepMind Green, add white border
            ax.hist(filtered_r2, bins=50, color=DM_GREEN, alpha=0.8, 
                    edgecolor='white', linewidth=0.5, zorder=3)
                        
            ax.set_xlabel("$R^2$", fontsize=25)
            ax.set_ylabel("Count", fontsize=25)
            
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "global_r2_dist.pdf"), format='pdf', bbox_inches='tight')
            plt.close()
            
            # Save global statistics to TXT
            global_mean = np.mean(filtered_r2)
            global_median = np.median(filtered_r2)
            global_std = np.std(filtered_r2)
            
            txt_path = os.path.join(OUTPUT_DIR, "global_r2_stats.txt")
            with open(txt_path, "w") as f:
                f.write("Global R2 Statistics\n")
                f.write("====================\n")
                f.write(f"Mean R2:   {global_mean:.6f}\n")
                f.write(f"Median R2: {global_median:.6f}\n")
                f.write(f"Std R2:    {global_std:.6f}\n")
                f.write(f"Total Params Sampled: {len(filtered_r2)}\n")
            print(f"Saved global statistics to {txt_path}")

def main():
    # 1. Initialization
    accelerator = Accelerator()
    set_seed(SEED)

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Starting distributed analysis on {accelerator.num_processes} GPUs.")
    
    # 2. Load Checkpoints
    all_state_dicts = load_all_checkpoints(accelerator)
    if not all_state_dicts:
        print("No checkpoints loaded.")
        return

    # 3. Run Analysis
    local_results = run_distributed_analysis(accelerator, all_state_dicts)

    # 4. Gather Results
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Gathering results from all processes...")

    gathered_results_nested = [None for _ in range(accelerator.num_processes)]
    if accelerator.num_processes > 1:
        dist.all_gather_object(gathered_results_nested, local_results)
    else:
        gathered_results_nested = [local_results]
    
    # 5. Aggregate and Save
    if accelerator.is_main_process:
        flat_results = []
        for item in gathered_results_nested:
            if isinstance(item, list):
                flat_results.extend(item)
            else:
                flat_results.append(item)
        
        if not flat_results:
            print("No results gathered. Exiting.")
            return

        # Plot summary
        layer_names_all = [r['name'] for r in flat_results]
        layer_r2_means_all = [r['mean_r2'] for r in flat_results]
        plot_heatmap_stats(layer_r2_means_all, layer_names_all)

        # Save statistics (CSV & TXT) and Global R2 Dist (DeepMind Style)
        save_statistics(flat_results)
        
        print(f"Done. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()