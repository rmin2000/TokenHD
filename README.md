# TokenHD: Code Release

This repository contains the core code for **TokenHD**, a pipeline for training
token-level hallucination detectors in large language models.

## Overview

TokenHD consists of four stages:

1. **Response generation** — sample LLM responses for math/STEM problems
2. **Hallucination annotation** — have critic models identify erroneous spans
3. **Ensemble label computation** — aggregate annotations from multiple critics into soft token-level labels
4. **Detector training** — fine-tune a token classifier on the soft labels with importance weighting

---

## Requirements

```bash
pip install -r requirements.txt
```

Set your OpenAI API key before running any API-based steps:

```bash
export OPENAI_API_KEY=your_key_here
```

---

## Pipeline

### Stage 1: Generate responses

```bash
python data_generation/generate_responses.py \
    --use_openai_key \
    --model gpt-4o-mini \
    --dataset_name hendrycks/competition_math \
    --folder_name math_train \
    --sampling_n 2 \
    --chunk 1 --tot_chunk 10
```

This saves JSONL files under `data/<model>/<folder_name>/`. Each entry contains
`problem`, `gt_answer`, `raw_answer`, and `correctness` (1 = exact match,
0 = mathematically equivalent, -1 = wrong).

For local inference with vLLM, omit `--use_openai_key` and pass a local model path.

---

### Stage 2: Annotate hallucinations

```bash
python annotation/annotate.py \
    --use_openai_key \
    --source_model_name gpt-4o-mini \
    --label_model_name gpt-4o-mini \
    --folder_name math_train \
    --rollout_num 3
```

This produces per-critic annotation files under
`data/<policy_model>/<folder_name>/verbal_labeler_<critic>/`.

---

### Stage 3: Restore annotated spans

Critics sometimes paraphrase rather than quote exactly. Run this to snap each
span to its exact substring in the original response:

```bash
python annotation/restore.py \
    --source_model_name gpt-4o-mini \
    --label_model_name gpt-4o-mini \
    --folder_name math_train \
    --restore_model gpt-4o-mini \
    --chunk 1
```

Output goes to `.../verbal_labeler_<critic>/restored/chunk_<n>.jsonl`.

---

### Stage 4: Compute ensemble labels

```bash
python ensemble/compute_ensemble_weights.py \
    --policy_model gpt-4o-mini \
    --folder_name math_train \
    --label_models "ModelA,ModelB,ModelC" \
    --tokenizer_name Qwen/Qwen3-8B \
    --weighted
```

Use `--weighted` to learn per-critic weights on a held-out validation subset
(adaptive ensemble). Without it, uniform averaging is used.
Output is saved to `data/<policy_model>/<folder_name>/ensemble/`.

---

### Stage 5: Train the detector

```bash
bash training/train.sh \
    Qwen3-1.7B \
    portion \
    1 \
    1.0 0.02 0.5 \
    ckpts/tokenhd-1.7b \
    data/gpt-4o-mini/math_train/ensemble/ensemble_weighted.jsonl
```

Arguments: `<model_size> <weighted_mode> <epochs> <incorrp> <corrp> <filtering_t> <output_dir> <data_files>`

- `weighted_mode`: `portion` (recommended) | `linear` | `log` | `none`
- `incorrp` / `corrp`: fraction of incorrect / correct samples to include
- `filtering_t`: minimum max soft-label value for a sample to qualify

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

Reports `S_incor` (token F1 on incorrect samples) and `S_cor` (recall on correct
samples), matching the metrics defined in the paper.

---

## Model Merging

To merge domain-specific detectors:

```bash
python model_merging/merge.py \
    --merge_method average_merging \
    --base_model Qwen/Qwen3-1.7B \
    --models_to_merge ckpts/tokenhd-math,ckpts/tokenhd-code \
    --output_dir ckpts/tokenhd-merged \
    --use_gpu
```

Supported methods: `average_merging`, `task_arithmetic`, `ties_merging`, `dare_merging`.

---

## Repository Structure

```
tokenhd_code/
├── data_generation/
│   └── generate_responses.py   # Stage 1: sample LLM responses
├── annotation/
│   ├── annotate.py             # Stage 2: critic annotation
│   └── restore.py              # Stage 3: snap spans to exact substrings
├── ensemble/
│   └── compute_ensemble_weights.py   # Stage 4: soft label construction
├── training/
│   ├── train.py                # Stage 5: importance-weighted token classifier training
│   ├── train.sh                # Multi-GPU launch script
│   └── fsdp_config.json        # FSDP configuration
├── evaluation/
│   └── evaluate.py             # Stage 6: S_incor / S_cor evaluation
├── model_merging/
│   ├── merge.py
│   ├── merging_methods.py
│   ├── task_vector.py
│   └── mask_weights_utils.py
├── prompts/
│   ├── judge.py                # Critic prompts for hallucination identification
│   ├── restore.txt             # Text span restoration prompt
│   └── correctness_check.txt  # Answer correctness judge prompt
├── utils.py                    # Shared utilities
├── optim_utils.py              # Ensemble weight optimizer
└── requirements.txt
```
