"""
Step 4 of the data pipeline: compute ensemble weights and produce token-level
training labels.

Given annotations from multiple critic models, this script either:
  (a) uniformly averages the annotations, or
  (b) learns optimal per-critic weights via gradient descent (adaptive ensemble).

The learned weights maximize label quality on a held-out validation subset.
Final output is a JSONL file where each entry contains token IDs and soft
label scores, ready for the training script.

Usage example:
    python ensemble/compute_ensemble_weights.py \
        --policy_model gpt-4o-mini \
        --folder_name math_train \
        --label_models "ModelA,ModelB,ModelC" \
        --weighted \
        --data_dir data \
        --output_dir data/gpt-4o-mini/math_train/ensemble
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import token_ensemble, token_ensemble_weighted, load_jsonl, hard_f1, data_preprocess
from optim_utils import WeightedVectorOptimizer

parser = argparse.ArgumentParser(description="Compute ensemble labels from multiple critic annotations.")
parser.add_argument("--policy_model", type=str, default="gpt-4o-mini")
parser.add_argument("--folder_name", type=str, default="math_train")
parser.add_argument("--label_models", type=str, required=True,
                    help="Comma-separated list of critic model names (matching folder names).")
parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-8B",
                    help="Tokenizer used to convert text spans to token indices.")
parser.add_argument("--weighted", action="store_true",
                    help="Use adaptive ensemble weighting (vs. uniform averaging).")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save the ensemble output JSONL.")
parser.add_argument("--val_fraction", type=float, default=0.1,
                    help="Fraction of training data used as the validation set for weight learning.")
args = parser.parse_args()

label_model_list = [m.strip() for m in args.label_models.split(",")]
output_dir = args.output_dir or f"{args.data_dir}/{args.policy_model}/{args.folder_name}/ensemble"
os.makedirs(output_dir, exist_ok=True)

label_list_all = data_preprocess(label_model_list, args.policy_model, args.folder_name, args.data_dir)
assert len(label_list_all) > 0, "No annotation data found."

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

# --- Compute per-model token-level soft labels ---
per_model_token_labels = []   # list[list[np.ndarray]]  shape: [num_models, num_samples]
raw_answers = []
problems = []
correctness_flags = []
for model_idx, label_list in enumerate(label_list_all):
    model_preds = []
    for item in label_list:
        raw_answer = item["raw_answer"]
        label_index_list = item["label_index_list"]
        correctness = item["correctness"]

        valid_labels = []
        for verbal_label in label_index_list:
            if verbal_label != [] and verbal_label != "No errors!":
                valid_labels.extend([x for x in verbal_label if x in raw_answer])

        _, token_weights = token_ensemble(
            raw_answer, valid_labels, len(valid_labels), tokenizer, correctness
        )
        model_preds.append(token_weights)

        if model_idx == 0:
            raw_answers.append(raw_answer)
            problems.append(item["problem"])
            correctness_flags.append(correctness)

    per_model_token_labels.append(model_preds)

num_samples = len(raw_answers)
print(f"Total samples: {num_samples}, Critics: {len(label_model_list)}")

# --- Learn ensemble weights on a validation subset ---
if args.weighted and len(label_list_all) > 1:
    val_size = max(1, int(num_samples * args.val_fraction))
    val_indices = list(range(val_size))
    train_indices = list(range(val_size, num_samples))

    val_input = [[per_model_token_labels[m][i] for i in val_indices] for m in range(len(label_list_all))]
    val_gt = [per_model_token_labels[0][i] for i in val_indices]  # use first model labels as proxy gt

    optimizer = WeightedVectorOptimizer(
        num_models=len(label_list_all),
        learning_rate=0.01,
        device="cpu",
        weighted_loss=True,
    )
    optimal_weights, _ = optimizer.train(val_input, val_gt, epochs=100, batch_size=64)
    print(f"Learned ensemble weights: {optimal_weights}")
else:
    optimal_weights = np.ones(len(label_list_all)) / len(label_list_all)
    train_indices = list(range(num_samples))
    print(f"Using uniform weights: {optimal_weights}")

# --- Apply weights and produce final dataset ---
dataset = []
for i in tqdm(train_indices, desc="Building dataset"):
    raw_answer = raw_answers[i]
    correctness = correctness_flags[i]

    all_spans = []
    all_weights = []
    for m_idx, label_list in enumerate(label_list_all):
        item = label_list[i]
        for verbal_label in item["label_index_list"]:
            if verbal_label != [] and verbal_label != "No errors!":
                for span in verbal_label:
                    if span in raw_answer:
                        all_spans.append(span)
                        all_weights.append(float(optimal_weights[m_idx]))

    if args.weighted and all_weights:
        token_ids, token_weights = token_ensemble_weighted(raw_answer, all_spans, all_weights, tokenizer)
    else:
        token_ids, token_weights = token_ensemble(
            raw_answer, all_spans, len(all_spans), tokenizer, correctness
        )

    dataset.append({
        "problem": problems[i],
        "raw_answer": raw_answer,
        "correctness": correctness,
        "token_ids": token_ids,
        "token_weights": token_weights.tolist(),
    })

suffix = "weighted" if args.weighted else "uniform"
output_file = os.path.join(output_dir, f"ensemble_{suffix}.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(dataset)} samples to {output_file}.")
