#!/bin/bash
# Launch TokenHD detector training with torchrun (multi-GPU).
#
# Usage:
#   bash training/train.sh <model_size> <weighted_mode> <epochs> <incorrp> <corrp> <filtering_t> <output_dir> <data_source>
#
# Arguments:
#   model_size     — Qwen3 model variant, e.g. Qwen3-1.7B or Qwen3-8B
#   weighted_mode  — token weighting strategy: portion | linear | log | none
#   epochs         — number of training epochs
#   incorrp        — incorrect sample ratio (float, e.g. 1.0)
#   corrp          — correct sample ratio relative to incorrect count (float, e.g. 0.02)
#   filtering_t    — minimum label threshold to include a sample (float, e.g. 0.5)
#   output_dir     — directory to save the trained checkpoint
#   data_source    — either:
#                      a local JSONL path:    data/tokenhd_math_train.jsonl
#                      or a HF dataset arg:   --hf_dataset mr233/TokenHD-training-data --hf_data_files tokenhd_math_train.jsonl
#
# Examples:
#   # From HuggingFace (no local data download needed):
#   bash training/train.sh Qwen3-1.7B portion 1 1.0 0.02 0.5 ckpts/tokenhd-1.7b \
#       "--hf_dataset mr233/TokenHD-training-data --hf_data_files tokenhd_math_train.jsonl"
#
#   # From local file:
#   bash training/train.sh Qwen3-1.7B portion 1 1.0 0.02 0.5 ckpts/tokenhd-1.7b \
#       data/tokenhd_math_train.jsonl

base_model="Qwen/$1"
weighted_mode=$2
epochs=$3
incorrp=$4
corrp=$5
filtering_t=$6
output_dir=$7
data_source=$8

# data_source can be a local path or HF args (--hf_dataset ... --hf_data_files ...)
if [[ "${data_source}" == --hf_* ]]; then
    data_arg="${data_source}"
else
    data_arg="--data_files=${data_source}"
fi

echo "Base model:    ${base_model}"
echo "Weighted mode: ${weighted_mode}"
echo "Epochs:        ${epochs}"
echo "Output dir:    ${output_dir}"

lr=1e-5
weight_decay=1e-4
micro_batch_size=1
gradient_accumulation_steps=1
gpu_count=$(nvidia-smi -L | wc -l)

torchrun --nproc-per-node ${gpu_count} --master_port 12465 \
    training/train.py \
    --block_size=20000 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    ${data_arg} \
    --model_name=${base_model} \
    --num_labels=1 \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="training/fsdp_config.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir=${output_dir} \
    --push_to_hub=false \
    --save_only_model=True \
    --weighted_mode=${weighted_mode} \
    --incorrect_ratio=${incorrp} \
    --correct_ratio=${corrp} \
    --filtering_t=${filtering_t} \
    --dataloader_num_workers=4 \
    --dataloader_pin_memory=True \
    --gradient_checkpointing=True
