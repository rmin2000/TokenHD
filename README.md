# Scalable Token-Level Hallucination Detection in Large Language Models

This repository contains the code for **TokenHD**, a pipeline for training lightweight token-level hallucination detectors in LLMs.

## Requirements

```bash
pip install -r requirements.txt
```

Set your OpenAI-compatible API key before running any API-based steps:

```bash
export OPENAI_API_KEY=your_key_here
```

---

## Pipeline

### Stage 1: Generate Responses

Sample LLM responses for each problem in a dataset and judge their correctness.

```bash
python data_generation/generate_responses.py \
    --use_openai_key \
    --model gpt-4o-mini \
    --judge_model gpt-4o-mini \
    --dataset_name hendrycks/competition_math \
    --split train \
    --folder_name math_train \
    --sampling_n 2 \
    --max_tokens 16000 \
    --chunk 1 --tot_chunk 10 \
    --data_dir data
```

| Argument | Description |
|---|---|
| `--use_openai_key` | Use the OpenAI API; omit to run local inference via vLLM |
| `--model` | Policy model (OpenAI model name or local HuggingFace path) |
| `--judge_model` | Model used to judge whether a predicted answer is correct |
| `--sampling_n` | Number of responses to sample per problem |
| `--chunk` / `--tot_chunk` | 1-indexed chunk index and total chunks for parallel processing |
| `--max_tokens` | Max generation tokens for the policy model |
| `--folder_name` | Sub-folder name to save results under `data/<model>/` |

Output is saved as JSONL under `data/<model>/<folder_name>/chunk_<n>.jsonl`. Each entry records `problem`, `gt_answer`, `raw_answer`, and `correctness` (1 = exact match, 0 = mathematically equivalent, −1 = wrong).

---

### Stage 2: Annotate Hallucinations

Prompt critic models to identify hallucinated text spans within each response.

```bash
python annotation/annotate.py \
    --use_openai_key \
    --source_model_name gpt-4o-mini \
    --label_model_name gpt-4o-mini \
    --folder_name math_train \
    --rollout_num 3 \
    --data_dir data
```

| Argument | Description |
|---|---|
| `--source_model_name` | Policy model whose responses are being annotated |
| `--label_model_name` | Critic model performing the annotation |
| `--rollout_num` | Number of annotation rollouts per response (averaged to produce soft labels) |

Output is saved under `data/<policy_model>/<folder_name>/verbal_labeler_<critic>/`.

---

### Stage 3: Restore Annotated Spans

Critics may paraphrase rather than quote exactly. This step snaps each annotated span to its verbatim substring in the original response.

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
| `--restore_model` | Model used to iteratively refine each span until it matches the original text |
| `--chunk` | Chunk index to process (matches the chunk from Stage 1) |

Output is written to `.../verbal_labeler_<critic>/restored/chunk_<n>.jsonl`.

---

### Stage 4: Compute Ensemble Labels

Aggregate annotations from multiple critics into a single soft token-level label sequence.

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
| `--label_models` | Comma-separated list of critic model names whose annotations are combined |
| `--tokenizer_name` | Tokenizer used to align text spans to token positions |
| `--weighted` | Learn per-critic weights via gradient descent (adaptive ensemble); omit for uniform averaging |

Output is saved to `data/<policy_model>/<folder_name>/ensemble/`.

---

### Stage 5: Train the Detector

Fine-tune a token classifier on the soft ensemble labels with importance weighting.

```bash
bash training/train.sh \
    Qwen3-1.7B \
    portion \
    1 \
    1.0 0.02 0.5 \
    ckpts/tokenhd-1.7b \
    data/gpt-4o-mini/math_train/ensemble/ensemble_weighted.jsonl
```

Positional arguments in order: `<model_size> <weighted_mode> <epochs> <incorrp> <corrp> <filtering_t> <output_dir> <data_files>`

| Argument | Description |
|---|---|
| `model_size` | Backbone size, e.g. `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-8B` |
| `weighted_mode` | How to translate soft labels into per-token loss weights: `portion` (recommended — weight proportional to label value), `linear`, `log`, or `none` (unweighted) |
| `epochs` | Number of training epochs |
| `incorrp` | Fraction of incorrect (hallucinated) samples to include in training |
| `corrp` | Fraction of correct (hallucination-free) samples to include in training |
| `filtering_t` | Minimum peak soft-label value a sample must have to qualify (filters out low-signal samples) |
| `output_dir` | Directory to save the trained checkpoint |
| `data_files` | Path to the ensemble JSONL file from Stage 4 |

---

### Stage 6: Evaluate

Evaluate a trained detector on annotated benchmark responses.

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
| `--policy_model` | Policy model that generated the evaluation responses |
| `--folder_name` | Benchmark name, matching the data sub-folder |
| `--annotator_models` | Comma-separated list of annotator models used to construct ground-truth labels |

Reports two metrics:
- **$S_\text{incor}$**: token-level F1 on hallucinated (incorrect) samples
- **$S_\text{cor}$**: recall on hallucination-free (correct) samples

---

## Model Merging

Train domain-specific detectors separately and merge their weights into a single generalist detector.

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
| `--merge_method` | Merging strategy: `average_merging`, `task_arithmetic`, `ties_merging`, `dare_merging` |
| `--base_model` | Base backbone model (used as the reference point for task-vector methods) |
| `--models_to_merge` | Comma-separated paths to the specialist checkpoints to merge |
| `--use_gpu` | Load models on GPU for faster merging |
