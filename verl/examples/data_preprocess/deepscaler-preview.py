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

import argparse
import os
import json
import datasets

from verl.utils.hdfs_io import copy, makedirs

DATA_PATH = "agentica-org/DeepScaleR-Preview-Dataset"
OUTPUT_DIR = "./verl/examples/data_preprocess/data"
INSTRUCTION_FOLLOWING = "Let\'s think step by step and output the final answer within \\boxed{}."


def extract_solution(solution_str):
    return solution_str


def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "lighteval/MATH"

    train_dataset = load_json(DATA_PATH)

    def make_map_fn(split):
        def process_fn(example, idx):
            print(example)
            question_raw = example.pop("problem")
            question = question_raw + " " + INSTRUCTION_FOLLOWING

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            print(data)
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(OUTPUT_DIR, "deepscaler-preview.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=OUTPUT_DIR, dst=hdfs_dir)