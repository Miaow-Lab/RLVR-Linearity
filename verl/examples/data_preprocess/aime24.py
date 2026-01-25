# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the AIME dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


DATA_PATH = "math-ai/aime24"
OUTPUT_DIR = "./verl/examples/data_preprocess/data"
INSTRUCTION_FOLLOWING = "Let\'s think step by step and output the final answer within \\boxed{}."


def extract_solution(solution_str):
    solution = re.search(r'\\boxed\{([^}]*)\}', solution_str)
    assert solution is not None
    final_solution = solution.group(1).strip()
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "aime24"

    dataset = datasets.load_dataset(DATA_PATH)
    test_dataset = dataset["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")
            question = question_raw + " " + INSTRUCTION_FOLLOWING

            answer_raw = example.pop("solution")
            solution = extract_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    keep_cols = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    cols_to_remove = [c for c in test_dataset.column_names if c not in keep_cols]
    test_dataset = test_dataset.remove_columns(cols_to_remove)

    test_dataset.to_parquet(os.path.join(OUTPUT_DIR, "aime24.parquet"))

    hdfs_dir = args.hdfs_dir
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=OUTPUT_DIR, dst=hdfs_dir)