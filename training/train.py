"""
Training script for the TokenHD token-level hallucination detector.

The detector is fine-tuned as a token classifier on top of a language model
backbone. Training uses soft labels (continuous scores in [0,1]) and an
importance-weighted loss that upweights hallucinated tokens to handle the
class imbalance between hallucinated and non-hallucinated tokens.

Key arguments:
    --model_name        HuggingFace model path (e.g. Qwen/Qwen3-1.7B)
    --data_files        Comma-separated paths to training JSONL files
    --output_dir        Directory to save the trained model
    --weighted_mode     Token weighting strategy: portion | linear | log | none
    --incorrect_ratio   Fraction of incorrect samples to include
    --correct_ratio     Fraction of correct samples (relative to incorrect count)
    --filtering_t       Minimum soft label value for a sample to be included

Example:
    torchrun --nproc-per-node 8 training/train.py \
        --model_name Qwen/Qwen3-1.7B \
        --data_files path/to/ensemble.jsonl \
        --output_dir ckpts/tokenhd-1.7b \
        --weighted_mode portion \
        --num_train_epochs 1 \
        --learning_rate 1e-5
"""

import os
import pickle
import hashlib
import logging
import random
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Union

import numpy as np
import torch
import transformers
import trl
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    PreTrainedTokenizerBase,
)
from transformers.utils import PaddingStrategy

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class SystemPromptDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        max_seq_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label_seq, attention_mask in zip(labels, attention_masks):
            full_label = [0.0] * max_seq_length
            active_positions = [j for j, mask in enumerate(attention_mask) if mask == 1]
            for pos_idx, label_val in enumerate(label_seq):
                if pos_idx < len(active_positions):
                    full_label[active_positions[pos_idx]] = label_val
            padded_labels.append(full_label)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.float)
        return batch


def data_split(training_dataset, incorrect_ratio=0.25, correct_ratio=0.8):
    random.shuffle(training_dataset)
    incorrect_data = [x for x in training_dataset if max(x["labels"]) != 0]
    correct_data = [x for x in training_dataset if max(x["labels"]) == 0]
    incorrect_data = incorrect_data[: int(len(incorrect_data) * incorrect_ratio)]
    correct_data = correct_data[: int(len(incorrect_data) * correct_ratio)]
    return incorrect_data + correct_data


def compute_batch_token_weights_portion(labels, attention_mask):
    batch_size = labels.shape[0]
    weights = torch.zeros_like(labels)
    for i in range(batch_size):
        valid_mask = attention_mask[i] == 1
        valid_labels = labels[i][valid_mask]
        pos_count = len(valid_labels[valid_labels > 0.5])
        neg_count = len(valid_labels[valid_labels <= 0.5])
        if pos_count == 0:
            w_pos = torch.tensor([1.0]).float().to(valid_labels.device)
            w_neg = torch.tensor([1.0]).float().to(valid_labels.device)
        else:
            total = pos_count + neg_count
            w_pos = torch.tensor([neg_count / total]).float().to(valid_labels.device)
            w_neg = torch.tensor([pos_count / total]).float().to(valid_labels.device)
        sample_weights = torch.where(valid_labels > 0.5, w_pos, w_neg)
        weights[i][valid_mask] = sample_weights
    return weights


def compute_batch_token_weights_linear(labels, attention_mask):
    weights = torch.ones_like(labels)
    for i in range(labels.shape[0]):
        valid_mask = attention_mask[i] == 1
        weights[i][valid_mask] = labels[i][valid_mask] * 9 + 1
    return weights


def compute_batch_token_weights_log(labels, attention_mask):
    weights = torch.ones_like(labels)
    for i in range(labels.shape[0]):
        valid_mask = attention_mask[i] == 1
        weights[i][valid_mask] = (
            torch.log(labels[i][valid_mask] + 1) / torch.log(torch.tensor(2.0)) * 9 + 1
        )
    return weights


class TokenLevelSoftLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=torch.ones_like(inputs["attention_mask"]),
        )
        token_labels = inputs["labels"].to(outputs.logits.device)
        attention_mask = inputs["attention_mask"].to(outputs.logits.device)
        logits = outputs.logits.squeeze(-1)

        mode = self.args.weighted_mode
        if mode == "linear":
            batch_weights = compute_batch_token_weights_linear(token_labels, attention_mask)
        elif mode == "log":
            batch_weights = compute_batch_token_weights_log(token_labels, attention_mask)
        elif mode == "portion":
            batch_weights = compute_batch_token_weights_portion(token_labels, attention_mask)
        elif mode == "none":
            batch_weights = torch.tensor([1.0], dtype=torch.float32).to(attention_mask.device)
        elif mode == "hard_portion":
            token_labels = (token_labels > 0.5).float()
            batch_weights = compute_batch_token_weights_portion(token_labels, attention_mask)
        elif mode == "hard_none":
            token_labels = (token_labels > 0.5).float()
            batch_weights = torch.tensor([1.0], dtype=torch.float32).to(attention_mask.device)
        else:
            raise ValueError(f"Unknown weighted_mode: {mode}")

        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, token_labels, reduction="none"
        )
        weighted_loss = bce_loss * batch_weights * attention_mask
        final_loss = weighted_loss.sum() / attention_mask.sum()

        if return_outputs:
            return final_loss, {"loss": final_loss}
        return final_loss


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen3-1.7B")
    block_size: int = field(default=10000)
    data_files: str = field(default="", metadata={"help": "Comma-separated paths to training JSONL files."})
    num_labels: int = field(default=1)
    weighted_mode: str = field(
        default="portion",
        metadata={"help": "Token weighting mode: portion | linear | log | none | hard_portion | hard_none."},
    )
    incorrect_ratio: float = field(default=1.0, metadata={"help": "Fraction of incorrect samples to keep."})
    correct_ratio: float = field(default=0.02, metadata={"help": "Fraction of correct samples relative to incorrect count."})
    filtering_t: float = field(default=0.5, metadata={"help": "Minimum max label value for a sample to be included."})
    max_train_samples: int = field(default=-1, metadata={"help": "Cap on training set size (random sample, seed=42)."})
    cache_dir: str = field(default="", metadata={"help": "Directory for tokenization cache. Defaults to system temp."})


def train():
    os.environ["WANDB_DISABLED"] = "true"
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    logging.info(f"Training config: {asdict(config)}")

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels, torch_dtype=torch.bfloat16
    )

    dataset_file_list = [p.strip() for p in config.data_files.split(",") if p.strip()]
    assert dataset_file_list, "No data files specified. Pass --data_files path1.jsonl,path2.jsonl"

    keep_columns = ["problem", "raw_answer", "correctness", "token_ids", "token_weights"]
    raw_dataset = load_dataset("json", data_files=dataset_file_list)
    raw_dataset = raw_dataset.select_columns([c for c in keep_columns if c in raw_dataset["train"].column_names])
    dataset = raw_dataset["train"].shuffle(seed=2345)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True, trust_remote_code=True)
    data_collator = SystemPromptDataCollator(tokenizer=tokenizer, padding=True)

    # Tokenization with optional disk cache
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        local_rank = 0

    cache_dir = config.cache_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "tokenhd_tokenize_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5((config.model_name + "|" + "|".join(dataset_file_list)).encode()).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"tok_{config.model_name.split('/')[-1]}_{cache_key}.pkl")

    all_tokenized = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                all_tokenized = pickle.load(fh)
            print(f"[tokenize-cache] HIT {cache_path} ({len(all_tokenized)} samples)")
        except Exception as e:
            print(f"[tokenize-cache] load failed ({e}), will retokenize.")
            all_tokenized = None

    if all_tokenized is None:
        print(f"[tokenize-cache] MISS — tokenizing {len(dataset)} samples...")
        all_tokenized = []
        for data in tqdm(dataset, desc="Tokenizing"):
            messages = [
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["raw_answer"]},
            ]
            result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            if "deepseek" in config.model_name.lower():
                input_ids = result[:-1]
            else:
                input_ids = result[:-2]
            labels = data["token_weights"]
            token_ids = data["token_ids"]
            attention_mask = [0] * (len(input_ids) - len(token_ids)) + [1] * len(token_ids)
            all_tokenized.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "token_ids": token_ids,
                "correctness": data["correctness"],
                "problem": data["problem"],
            })
        if local_rank == 0:
            tmp_path = cache_path + f".tmp.{os.getpid()}"
            try:
                with open(tmp_path, "wb") as fh:
                    pickle.dump(all_tokenized, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.rename(tmp_path, cache_path)
                print(f"[tokenize-cache] wrote {cache_path} ({len(all_tokenized)} samples)")
            except Exception as e:
                print(f"[tokenize-cache] write failed: {e}")

    correct_data_list, incorrect_data_list = [], []
    correct_id_dict, incorrect_id_dict = {}, {}

    for item in tqdm(all_tokenized, desc="Filtering"):
        labels = item["labels"]
        attention_mask = item["attention_mask"]
        input_ids = item["input_ids"]
        correctness = item["correctness"]
        problem = item["problem"]
        token_ids = item["token_ids"]

        if len([l for l in labels if l > 0.5]) / len(labels) > 0.5:
            continue
        if correctness in [0, 1] and sum(labels[-20:]) != 0:
            continue

        if max(labels) == 0 and correctness in [0, 1]:
            if correct_id_dict.get(problem, 0) >= 2:
                continue
            correct_id_dict[problem] = correct_id_dict.get(problem, 0) + 1
            correct_data_list.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "token_ids": token_ids,
            })

        if max(labels) > config.filtering_t and correctness == -1:
            if incorrect_id_dict.get(problem, 0) >= 2:
                continue
            incorrect_id_dict[problem] = incorrect_id_dict.get(problem, 0) + 1
            incorrect_data_list.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "token_ids": token_ids,
            })

    training_data_list = data_split(
        correct_data_list + incorrect_data_list,
        incorrect_ratio=config.incorrect_ratio,
        correct_ratio=config.correct_ratio,
    )

    if config.max_train_samples > 0 and len(training_data_list) > config.max_train_samples:
        rng = random.Random(42)
        training_data_list = rng.sample(training_data_list, config.max_train_samples)
        print(f"[max_train_samples] subsampled to {len(training_data_list)}")

    training_data = Dataset.from_list(training_data_list)
    n_incor = len([x for x in training_data if max(x["labels"]) != 0])
    n_cor = len([x for x in training_data if max(x["labels"]) == 0])
    print(f"Training set — incorrect: {n_incor}, correct: {n_cor}")

    args.max_seq_length = config.block_size
    args.weighted_mode = config.weighted_mode
    trainer = TokenLevelSoftLabelTrainer(
        model=model,
        train_dataset=training_data,
        eval_dataset=training_data,
        args=args,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
