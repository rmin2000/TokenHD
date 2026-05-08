"""
Evaluate a trained TokenHD detector on annotated benchmarks.

The script loads ground-truth token-level hallucination annotations from
multiple critic models, aggregates them into soft labels, runs the detector,
and reports S_incor (token F1 on incorrect samples) and S_cor (recall on
correct samples).

Usage:
    python evaluation/evaluate.py \
        --model_path ckpts/tokenhd-1.7b \
        --policy_model gpt-4o-mini \
        --folder_name math_500 \
        --annotator_models "ModelA,ModelB" \
        --data_dir data
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from utils import load_jsonl, hard_f1, extract_answer_from_box, token_ensemble, data_preprocess

parser = argparse.ArgumentParser(description="Evaluate TokenHD detector.")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the trained TokenHD checkpoint.")
parser.add_argument("--policy_model", type=str, default="gpt-4o-mini",
                    help="Policy model that generated the evaluation responses.")
parser.add_argument("--folder_name", type=str, default="math_500",
                    help="Benchmark name (matches the data sub-folder).")
parser.add_argument("--annotator_models", type=str, required=True,
                    help="Comma-separated list of annotator model names used for ground-truth labels.")
parser.add_argument("--data_dir", type=str, default="data",
                    help="Root directory for evaluation data.")
args = parser.parse_args()

label_model_list = [m.strip() for m in args.annotator_models.split(",")]
label_list_all = data_preprocess(label_model_list, args.policy_model, args.folder_name, args.data_dir)

np.random.seed(2333)
for i in range(len(label_list_all)):
    np.random.shuffle(label_list_all[i])

model = AutoModelForTokenClassification.from_pretrained(
    args.model_path, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

precision_incor, recall_incor, f1_incor = [], [], []
precision_cor, recall_cor, f1_cor = [], [], []
mse_loss_list = []
length_list = []
correct_count = 0
incorrect_count = 0

correct_question_dict = {}
incorrect_question_dict = {}

for tmp_idx, item in tqdm(enumerate(label_list_all[0])):
    problem = item["problem"]
    raw_answer = item["raw_answer"]
    label_data = [x[tmp_idx] for x in label_list_all]

    new_label_list = []
    valid_count = 0
    valid_count_incorrect = 0

    for label in label_data:
        for verbal_label in label["label_index_list"]:
            if verbal_label != []:
                valid_count += 1
            if verbal_label not in [[], "No errors!"]:
                valid_count_incorrect += 1
                new_label_list.extend([x for x in verbal_label if x in raw_answer])

    if item["correctness"] == -1:
        if valid_count_incorrect == 0:
            continue
        token_ids, token_weights_gt, start_id = token_ensemble(
            raw_answer, new_label_list, valid_count_incorrect, tokenizer, item["correctness"], return_suffix=True
        )
        if start_id != -1 and max(token_weights_gt[: max(start_id - 2, 0)]) <= 0.9:
            continue
    else:
        token_ids, token_weights_gt, start_id = token_ensemble(
            raw_answer, new_label_list, valid_count, tokenizer, item["correctness"], return_suffix=True
        )

    token_weights_gt_hard = np.array([1 if l > 0.5 else 0 for l in token_weights_gt], dtype=np.float32)

    if item["correctness"] in [0, 1] and max(token_weights_gt_hard) == 0:
        cor_flag = True
        correct_count += 1
        if correct_question_dict.get(problem, 0) >= 4:
            continue
        correct_question_dict[problem] = correct_question_dict.get(problem, 0) + 1

    elif item["correctness"] == -1 and max(token_weights_gt_hard) == 1:
        cor_flag = False
        incorrect_count += 1
        if extract_answer_from_box(raw_answer) is None and "math" in args.data_dir:
            continue
        if incorrect_question_dict.get(problem, 0) >= 4:
            continue
        incorrect_question_dict[problem] = incorrect_question_dict.get(problem, 0) + 1
    else:
        continue

    messages = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": raw_answer},
    ]
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if "deepseek" in args.model_path.lower():
        input_ids = result[:-1]
    else:
        input_ids = result[:-2]

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=model.device).reshape(1, -1)
    with torch.no_grad():
        outputs = model(input_ids=input_tensor)
    sigmoid_logits = torch.sigmoid(outputs.logits.view(-1)[-len(token_weights_gt_hard):]).float().cpu().numpy()
    token_weights_pred = np.array(sigmoid_logits, dtype=np.float32)
    token_weights_pred_hard = np.array([1 if l > 0.5 else 0 for l in token_weights_pred], dtype=np.float32)

    length_list.append(len(token_weights_gt))

    del result, outputs, input_tensor

    precision, recall, f1 = hard_f1(token_weights_gt_hard, token_weights_pred_hard)
    if cor_flag:
        precision_cor.append(precision)
        recall_cor.append(recall)
        f1_cor.append(f1)
    else:
        precision_incor.append(precision)
        recall_incor.append(recall)
        f1_incor.append(f1)

    mse_loss_list.append(np.mean((token_weights_gt - token_weights_pred) ** 2))

print("=" * 80)
print(f"Model:          {args.model_path}")
print(f"Benchmark:      {args.folder_name}  |  Policy: {args.policy_model}")
print(f"Incorrect:      Prec={np.mean(precision_incor)*100:.2f}  Rec={np.mean(recall_incor)*100:.2f}  F1={np.mean(f1_incor)*100:.2f}  (n={incorrect_count})")
if correct_count > 0:
    print(f"Correct:        Prec={np.mean(precision_cor)*100:.2f}  Rec={np.mean(recall_cor)*100:.2f}  F1={np.mean(f1_cor)*100:.2f}  (n={correct_count})")
print(f"Avg length:     {np.mean(length_list):.1f} tokens")
print("=" * 80)

os.makedirs(os.path.join(args.data_dir, args.policy_model, args.folder_name, "results"), exist_ok=True)
results = {
    "model_path": args.model_path,
    "benchmark": args.folder_name,
    "policy_model": args.policy_model,
    "precision_incorrect": float(np.mean(precision_incor) * 100) if precision_incor else 0,
    "recall_incorrect": float(np.mean(recall_incor) * 100) if recall_incor else 0,
    "f1_incorrect": float(np.mean(f1_incor) * 100) if f1_incor else 0,
    "precision_correct": float(np.mean(precision_cor) * 100) if precision_cor else 0,
    "recall_correct": float(np.mean(recall_cor) * 100) if recall_cor else 0,
    "f1_correct": float(np.mean(f1_cor) * 100) if f1_cor else 0,
    "incorrect_count": incorrect_count,
    "correct_count": correct_count,
}
out_file = os.path.join(
    args.data_dir, args.policy_model, args.folder_name, "results",
    args.model_path.split("/")[-1] + ".json"
)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {out_file}")

torch.cuda.empty_cache()
