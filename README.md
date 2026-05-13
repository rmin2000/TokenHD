# Scalable Token-Level Hallucination Detection in Large Language Models

Code for **TokenHD**, a pipeline for training token-level hallucination detectors in LLMs.

## Resources

- **Pre-trained Models**: [HuggingFace Collection](https://huggingface.co/collections/mr233/tokenhd-6a0432e8a29a1e2a01e7fd7b)
- **Training Data**: [mr233/TokenHD-training-data](https://huggingface.co/datasets/mr233/TokenHD-training-data) — 82k math + 42k code samples with token-level soft labels

## Pre-trained Models

| Model | HuggingFace | Training Domain |
|---|---|---|
| TokenHD-0.6B | [mr233/TokenHD-0.6B](https://huggingface.co/mr233/TokenHD-0.6B) | Mathematics |
| TokenHD-1.7B | [mr233/TokenHD-1.7B](https://huggingface.co/mr233/TokenHD-1.7B) | Mathematics |
| TokenHD-4B | [mr233/TokenHD-4B](https://huggingface.co/mr233/TokenHD-4B) | Mathematics |
| TokenHD-8B | [mr233/TokenHD-8B](https://huggingface.co/mr233/TokenHD-8B) | Mathematics |
| TokenHD-8B-Mix | [mr233/TokenHD-8B-Mix](https://huggingface.co/mr233/TokenHD-8B-Mix) | Mathematics + Code |

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_id = "mr233/TokenHD-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=1)
model.eval()

problem = "What is the capital of France?"
response = "The capital of France is London."

messages = [
    {"role": "user", "content": problem},
    {"role": "assistant", "content": response},
]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)[:-2]
input_tensor = torch.tensor(input_ids).unsqueeze(0)

with torch.no_grad():
    logits = model(input_ids=input_tensor).logits  # shape: (1, seq_len, 1)

# scores for response tokens only (last len(response_tokens) positions)
response_ids = tokenizer.encode(response, add_special_tokens=False)
scores = torch.sigmoid(logits.squeeze(-1).squeeze(0))[-len(response_ids):]
# scores[i] is the hallucination probability for the i-th response token
```

---

## Requirements

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

---

## Pipeline

### Stage 1: Generate Responses

```bash
python data_generation/generate_responses.py \
    --use_openai_key \
    --model gpt-4o-mini \
    --judge_model gpt-4o-mini \
    --dataset_name hendrycks/competition_math \
    --split train \
    --folder_name math_train \
    --sampling_n 2 \
    --chunk 1 --tot_chunk 10 \
    --data_dir data
```

| Argument | Description |
|---|---|
| `--use_openai_key` | Use the OpenAI API; omit to run local inference via vLLM |
| `--model` | Policy model (OpenAI model name or HuggingFace path for vLLM) |
| `--judge_model` | Model used to judge answer correctness |
| `--sampling_n` | Number of responses sampled per problem |
| `--chunk` / `--tot_chunk` | 1-indexed chunk index and total number of chunks (for parallel runs) |

Output: `data/<model>/<folder_name>/chunk_<n>.jsonl`. Each entry has `problem`, `gt_answer`, `raw_answer`, and `correctness` (1 = exact match, 0 = equivalent, −1 = wrong).

Before Stage 2, merge all chunks:

```bash
cat data/gpt-4o-mini/math_train/chunk_*.jsonl > data/gpt-4o-mini/math_train/all.jsonl
```

---

### Stage 2: Annotate Hallucinations

```bash
python annotation/annotate.py \
    --use_openai_key \
    --source_model_name gpt-4o-mini \
    --label_model_name gpt-4o-mini \
    --folder_name math_train \
    --rollout_num 3 \
    --chunk 1 --tot_chunk 10 \
    --data_dir data
```

| Argument | Description |
|---|---|
| `--label_model_name` | Critic model that identifies hallucinated spans |
| `--rollout_num` | Annotation rollouts per response; results are averaged into soft labels |
| `--error_only` | Only annotate responses with `correctness == -1` |
| `--chunk` / `--tot_chunk` | Split annotation work across parallel jobs |

Output: `data/<policy_model>/<folder_name>/verbal_labeler_<critic>/chunk_<n>.jsonl`.

---

### Stage 3: Restore Annotated Spans

Critics sometimes paraphrase rather than quote exactly. This step iteratively snaps each span to its verbatim substring in the original response.

```bash
python annotation/restore.py \
    --source_model_name gpt-4o-mini \
    --label_model_name gpt-4o-mini \
    --folder_name math_train \
    --restore_model gpt-4o-mini \
    --chunk 1 \
    --data_dir data
```

| Argument | Description |
|---|---|
| `--restore_model` | Model used to locate the best-matching exact substring |

Output: `.../verbal_labeler_<critic>/restored/chunk_<n>.jsonl`.

---

### Stage 4: Compute Ensemble Labels

```bash
python ensemble/compute_ensemble_weights.py \
    --policy_model gpt-4o-mini \
    --folder_name math_train \
    --label_models "ModelA,ModelB,ModelC" \
    --tokenizer_name Qwen/Qwen3-8B \
    --weighted \
    --data_dir data
```

| Argument | Description |
|---|---|
| `--label_models` | Comma-separated critic model names whose annotations are combined |
| `--tokenizer_name` | Tokenizer used to align text spans to token positions |
| `--weighted` | Learn per-critic weights via gradient descent (adaptive ensemble); omit for uniform averaging |
| `--val_fraction` | Fraction of data held out to learn ensemble weights (default: 0.1) |

Output: `data/<policy_model>/<folder_name>/ensemble/`.

---

### Stage 5: Train the Detector

```bash
bash training/train.sh \
    Qwen3-1.7B \
    portion \
    1 \
    1.0 0.02 0.5 \
    ckpts/tokenhd-1.7b \
    data/gpt-4o-mini/math_train/ensemble/ensemble_weighted.jsonl
```

Positional arguments: `<model_size> <weighted_mode> <epochs> <incorrp> <corrp> <filtering_t> <output_dir> <data_files>`

| Argument | Description |
|---|---|
| `model_size` | Qwen3 backbone variant, e.g. `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-8B` |
| `weighted_mode` | Per-token loss weighting: `portion` (weight ∝ soft label value), `linear`, `log`, or `none` |
| `epochs` | Number of training epochs |
| `incorrp` | Fraction of incorrect (hallucinated) samples to retain |
| `corrp` | Size of the correct set as a multiple of the retained incorrect set (e.g. `0.02` keeps 2% as many correct samples as incorrect) |
| `filtering_t` | A sample is excluded if its maximum soft label value is below this threshold |

The script uses `lr=1e-5`, `weight_decay=1e-4`, and `gradient_checkpointing=True` by default. Edit `training/train.sh` to change these.

---

### Stage 6: Evaluate

```bash
python evaluation/evaluate.py \
    --model_path ckpts/tokenhd-1.7b \
    --policy_model gpt-4o-mini \
    --folder_name math_500 \
    --annotator_models "ModelA,ModelB" \
    --data_dir data
```

| Argument | Description |
|---|---|
| `--model_path` | Path to the trained TokenHD checkpoint |
| `--annotator_models` | Comma-separated annotator models used to build ground-truth labels |

Reports **S_incor** (token F1 on hallucinated samples) and **S_cor** (recall on hallucination-free samples).

---

## Model Merging

Train domain-specific detectors separately and merge their weights.

```bash
python model_merging/merge.py \
    --merge_method average_merging \
    --base_model Qwen/Qwen3-1.7B \
    --models_to_merge ckpts/tokenhd-math,ckpts/tokenhd-code \
    --output_dir ckpts/tokenhd-merged \
    --use_gpu
```

| Argument | Description |
|---|---|
| `--merge_method` | `average_merging`, `task_arithmetic`, `ties_merging`, `ties_merging_dare` |
| `--base_model` | Reference backbone for task-vector-based methods (`task_arithmetic`, `ties_merging`, `ties_merging_dare`); not required for `average_merging` |
| `--models_to_merge` | Comma-separated paths to specialist checkpoints |
