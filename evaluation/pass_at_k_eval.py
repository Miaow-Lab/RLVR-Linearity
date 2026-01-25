import json
from math import comb
from prm800k.grading import grader
import re
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np

INFER_PATH = "evaluation/outputs/aime24_distill-qwen-1-5b.json"
OUTPUT_DIR = "./outputs"
K = 64


def load_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def find_braces_content(s, tag='\\boxed'):
    idx = s.find(tag+'{')
    if idx == -1:
        return None
    start = idx + len(tag) + 1
    count = 1
    i = start
    while i < len(s) and count:
        if s[i] == '{':
            count += 1
        elif s[i] == '}':
            count -= 1
        i += 1
    # Extract content
    if count == 0:
        return s[start:i-1]
    else:
        return None


def extract_final_answer(output: str) -> str:
    v = find_braces_content(output, '\\boxed')
    if v:
        return v.strip()
    v = find_braces_content(output, '\\fbox')
    if v:
        return v.strip()
    m_frac = re.findall(r'\\[d]?frac\s*{[^{}]*}\s*{[^{}]*}', output)
    if m_frac:
        return m_frac[-1].strip()
    m_all = re.findall(r'\$([^\$]+)\$', output, re.DOTALL)
    if m_all:
        return m_all[-1].strip()
    m_all = re.findall(r'\d+/\d+', output)
    if m_all:
        return m_all[-1]
    lines = [l.strip(' $\n\r') for l in output.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return output.strip()


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k
    n: total number of generations
    c: number of correct generations
    k: number of generations to select
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    all_responses = load_json(INFER_PATH)
    
    k_values = [1, 4, 8, 16, 32, 64]
    all_pass_k = defaultdict(list)
    all_avg_k = defaultdict(list)  # Store Avg@k results

    for i, responses in enumerate(tqdm(all_responses)):
        problem = responses['problem']
        gt = responses['gt']

        correct_cnt = 0
        total_cnt = len(responses['generated_texts'])
        for res in responses['generated_texts']:
            answer = extract_final_answer(res['text'])
            if grader.grade_answer(answer, gt):
                correct_cnt += 1
        
        for k in k_values:
            if k <= total_cnt:
                # Calculate Pass@k
                pass_k = calculate_pass_at_k(total_cnt, correct_cnt, k)
                all_pass_k[k].append(pass_k)
                
                # Calculate Avg@k
                avg_k = correct_cnt / total_cnt if total_cnt > 0 else 0
                all_avg_k[k].append(avg_k)
    
    # Print results
    print(f"\nModel Evaluation Results (Total Problems: {len(all_responses)}):")
    print("-" * 50)
    
    # Determine the maximum k in the list to identify when to print Avg@k
    max_k = max(k_values) if k_values else 0

    for k in k_values:
        if k in all_pass_k:
            pass_at_k = np.mean(all_pass_k[k])
            
            if k == max_k:
                # Only print Avg@k for the largest k
                avg_at_k = np.mean(all_avg_k[k])
                print(f"  k={k:<2}: Pass@k = {pass_at_k:.4f} | Avg@k = {avg_at_k:.4f}")
            else:
                # For other k, only print Pass@k
                print(f"  k={k:<2}: Pass@k = {pass_at_k:.4f}")
                
    print("-" * 50)


if __name__ == "__main__":
    main()