<div align="center">

# Not All Steps are Informative: <br> On the Linearity of LLMs’ RLVR Training

[![Paper](https://img.shields.io/badge/arXiv-A82F27?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2601.04537v2)
[![Dataset](https://img.shields.io/badge/Datasets-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Miaow-Lab/RLVR-Linearity-Dataset)
[![Weights](https://img.shields.io/badge/Weights-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Miaow-Lab/RLVR-Linearity-Checkpoints)

<!-- Optional: Add License or Python version badges here -->
<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) -->

</div>

> [!IMPORTANT]
> **🌟 If you find this repository useful, please consider giving it a star!**
> 
> **🔥 News**
> - **[2026/01]** We have released the full codebase, including linearity analysis, RL training on `verl`, acceleration methods, and evaluation scripts. Preprocessed RL [datasets](https://huggingface.co/datasets/Miaow-Lab/RLVR-Linearity-Dataset) and [checkpoints](https://huggingface.co/Miaow-Lab/RLVR-Linearity-Checkpoints) are now available.

This repository contains the official implementation of the paper **"Not All Steps are Informative: On the Linearity of LLMs’ RLVR Training"**.

We reveal a critical phenomenon: **during RLVR (Reinforcement Learning with Verification Rewards), LLMs evolve in a remarkably linear manner.** Leveraging this observation, we demonstrate that future model states can be accurately predicted from intermediate checkpoints via extrapolation, effectively bypassing expensive training steps.

<p align="center">
<img src="./assets/method.png" width="85%" alt="Overview" />
</p>

## 📊 Linearity Analysis

<p align="center">
<img src="./assets/weight_token-logprob_r2.png" width="95%" alt="weight_token-logprob_r2" />
</p>

**Figure 1: Linearity of model weights and outputs during RLVR training.**
(a) and (b) show the distribution of $R^2$ scores for weights and token log-probabilities, respectively. Both distributions are strongly concentrated around 0.9, indicating high linearity.
(c) plots the trajectories of four randomly selected weights, while (d) tracks token log-probability shifts at four example positions.

<p align="center">
<img src="./assets/r2_generalization_analysis.png" width="95%" alt="r2_generalization_analysis" />
</p>

**Figure 2: Consistency across diverse setups.**
Linearity remains robust across various settings. $R^2$ scores consistently exceed 0.7 (dashed line) regardless of the base model (e.g., DS-Qwen, DS-Llama), model scale (1.5B to 8B), or training algorithm (GSPO, Reinforce++, and GRPO).

## 🚀 Accelerating RLVR via Extrapolation

Building on the linearity of RLVR training, we propose **Logits Extrapolation**, **Weight Extrapolation**, and **RL-Extra**. These methods enable the prediction of model behavior at future steps using early trajectories, significantly accelerating the training process.

<p align="center">
<img src="./assets/experiments_overall.png" width="95%" alt="Experimental Results" />
</p>

**Key Findings:**
- **Logits Extrapolation:** Delivers consistent accuracy improvements over standard RL on benchmarks like AIME and LiveCodeBench (LCB).
- **Weight Extrapolation:** Achieves high fidelity in predicting future weights, particularly on AIME24.
- **Efficiency:** Our methods significantly reduce the number of actual training steps required to reach target accuracy.
- **RL-Extra vs. GRPO:** With a fixed training budget (actual steps $s$), RL-Extra consistently outperforms the GRPO baseline across AIME24, AIME25, MATH500, and LiveCodeBench.

## 🛠️ Usage

### 1. Installation
Install the `verl` environment (ensure you are in the project root):

```bash
cd verl
pip3 install -e .[vllm]
```

### 2. Data Preparation
We utilize **DeepSeek-R1-Distill-Qwen-1.5B** as the base model.
1.  **Generate Responses:** We generated 64 responses for each AIME24 query. 
    *   Path: `evaluation/outputs/aime24_distill-qwen-1-5b.json`
2.  **Preprocessing:** Use the provided scripts to format data for RL training and evaluation within the `verl` framework.
    *   Script location: `verl/examples/data_preprocess`

> **Note:** The processed dataset is readily available on [HuggingFace Dataset](https://huggingface.co/datasets/Miaow-Lab/RLVR-Linearity-Dataset).

### 3. RL Training (Baseline)
To reproduce the DeepScaleR baseline (using GRPO), run the following command:

```bash
bash verl/examples/grpo_trainer/run_distill-qwen-1-5b_deepscaler.sh
```

<!-- > **Note:** Pre-trained RL checkpoints are available on [HuggingFace Models](https://huggingface.co/datasets/Miaow-Lab/RLVR-Linearity-Checkpoints). -->

### 4. Linearity Analysis
**Model Outputs (Token Log-probs):**
Use previously generated responses as probes to compute conditional log-probabilities for each token across checkpoints.

```bash
# 1. Compute log-probabilities
bash scripts/run_token_logprob_linearity.sh

# 2. Plot R^2 distribution
python3 analysis/token_logprob/plot_token_logprob_linearity.py
```

**Model Weights:**
Perform linear regression on model weights across training steps.

```bash
bash scripts/run_weight_linearity.sh
```

### 5. Extrapolation Methods & RL-Extra

| Method | Description | Command |
| :--- | :--- | :--- |
| **Logits Extrapolation** | Extrapolates logits to improve performance. | `bash scripts/run_logits_extrapolation.sh` |
| **Weight Extrapolation** | Extrapolates weights to accelerate RLVR. | `bash scripts/run_weight_extrapolation.sh` |
| **RL-Extra** | Corrects gradient trajectory for efficiency. | `bash scripts/run_rl_extrapolation.sh` |

### 6. Evaluation
**Inference (vLLM):**
```bash
python evaluation/inference_vllm_offline.py \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --data aime24
```

**Metrics Calculation (pass@k, avg@k):**
```bash
python evaluation/pass_at_k_eval.py
```

## ✉️ Contact

For questions or feedback, please contact [Tianle Wang](mailto:louis.wng@outlook.com).

## 🖊️ Citation

If you find this work helpful, please cite our paper:

```bibtex
@misc{wang2026stepsinformativelinearityllms,
      title={Not All Steps are Informative: On the Linearity of LLMs' RLVR Training}, 
      author={Tianle Wang and Zhongyuan Wu and Shenghao Jin and Hao Xu and Wei Chen and Ning Miao},
      year={2026},
      eprint={2601.04537},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.04537}, 
}
```
