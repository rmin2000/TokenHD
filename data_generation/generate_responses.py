"""
Step 1 of the data pipeline: generate LLM responses for math/STEM problems.

For each problem in the dataset, sample `--sampling_n` responses using either
a local model (via vLLM) or the OpenAI API. Each response is judged for
correctness against the ground-truth answer.

Required environment variable:
    OPENAI_API_KEY  — set this before running if using --use_openai_key

Output format (JSONL, one entry per response):
    {
        "idx": int,
        "problem": str,
        "gt_answer": str,
        "pred_answer": str | null,
        "raw_answer": str,
        "correctness": int   # 1 = same as gt, 0 = mathematically equivalent, -1 = wrong
    }
"""

import os
import json
import argparse
import torch

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils import load_jsonl, get_completion, extract_answer_from_box
from prompts.correctness_check import CORRECTNESS_CHECK_PROMPT

parser = argparse.ArgumentParser(description="Generate LLM responses for math/STEM benchmarks.")
parser.add_argument("--use_openai_key", action="store_true",
                    help="Use OpenAI API instead of local vLLM inference.")
parser.add_argument("--model", type=str, default="gpt-4o-mini",
                    help="Model name (HuggingFace path for vLLM, or OpenAI model name).")
parser.add_argument("--judge_model", type=str, default="gpt-4o-mini",
                    help="Model used to judge answer correctness.")
parser.add_argument("--chunk", type=int, default=1,
                    help="1-indexed chunk index (must be >= 1).")
parser.add_argument("--tot_chunk", type=int, default=1,
                    help="Total number of chunks.")
parser.add_argument("--max_tokens_judge", type=int, default=16000)
parser.add_argument("--max_tokens", type=int, default=16000)
parser.add_argument("--folder_name", type=str, required=True,
                    help="Sub-folder name under data_dir/<model>/ to save results.")
parser.add_argument("--sampling_n", type=int, default=2,
                    help="Number of responses to sample per problem.")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--dataset_name", type=str, default="hendrycks/competition_math",
                    help="HuggingFace dataset name.")
parser.add_argument("--pipeline_parallel_size", type=int, default=2)
parser.add_argument("--data_dir", type=str, default="data",
                    help="Root directory for saving generated data.")
args = parser.parse_args()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def load_benchmark(dataset_name, split):
    parsed_data = []

    if dataset_name == "Idavidrein/gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        for item in dataset:
            parsed_data.append({
                "problem": item["Question"],
                "gt_answer": item["Correct Answer"],
            })

    elif dataset_name == "Hothan/OlympiadBench-Math":
        import os as _os
        parquet_path = _os.path.join("data", "olym_bench", "OE_TO_maths_en_COMP.parquet")
        dataset = load_dataset("parquet", data_files=parquet_path)["train"]
        for item in dataset:
            parsed_data.append({
                "problem": item["question"],
                "gt_answer": ", ".join(item["final_answer"]),
            })

    elif dataset_name == "TheFinAI/Fino1_Reasoning_Path_FinQA":
        dataset = load_dataset(dataset_name, split=split)
        for item in dataset:
            parsed_data.append({
                "problem": item["Open-ended Verifiable Question"],
                "gt_answer": item["Ground-True Answer"],
            })

    else:
        dataset = load_dataset(dataset_name, split=split)
        for item in dataset:
            parsed_data.append({k: item[k] for k in item.keys()})

    return parsed_data


ori_data_list = load_benchmark(dataset_name=args.dataset_name, split=args.split)
print(f"Loaded {len(ori_data_list)} items.")

assert args.chunk >= 1, "Chunk index must start from 1."
cur_chunk = args.chunk - 1
if cur_chunk == args.tot_chunk - 1:
    data_list = ori_data_list[cur_chunk * len(ori_data_list) // args.tot_chunk :]
else:
    data_list = ori_data_list[
        cur_chunk * len(ori_data_list) // args.tot_chunk :
        (cur_chunk + 1) * len(ori_data_list) // args.tot_chunk
    ]

print(f"Chunk {args.chunk}/{args.tot_chunk}: {len(data_list)} problems.")

save_file = f"{args.data_dir}/{args.model.split('/')[-1]}/{args.folder_name}/chunk_{args.chunk}.jsonl"
os.makedirs(os.path.dirname(save_file), exist_ok=True)

if os.path.exists(save_file):
    start_index = len(load_jsonl(save_file)) // args.sampling_n
    writing_var = "a"
else:
    start_index = 0
    writing_var = "w"

if not args.use_openai_key:
    print("Using vLLM for local inference.")
    available_gpus = torch.cuda.device_count()
    assert available_gpus > 0, "No GPUs available."
    llm = LLM(
        model=args.model,
        tensor_parallel_size=available_gpus // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        n=1,
        skip_special_tokens=False,
        temperature=0.6,
        top_p=0.95,
    )

inst = "Please reason step by step, and put your final answer within \\boxed{}."

with open(save_file, writing_var, encoding="utf-8") as f:
    for idx in range(len(data_list)):
        print(f"Processing {idx + 1}/{len(data_list)} | {args.model} | Chunk {args.chunk}")
        if idx < start_index:
            continue

        problem = data_list[idx].get("problem") or data_list[idx].get("Question", "")
        gt_answer = data_list[idx].get("gt_answer") or data_list[idx].get("answer", "")

        for _ in range(args.sampling_n):
            if args.use_openai_key:
                resp = get_completion(
                    [{"role": "user", "content": problem.strip() + "\n" + inst}],
                    model=args.model,
                    max_tokens=args.max_tokens,
                    client=client,
                )
                if isinstance(resp, str):
                    raw_answer = resp
                elif resp is not None:
                    raw_answer = resp.choices[0].message.content
                else:
                    raw_answer = ""
            else:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": problem.strip() + "\n" + inst}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                out = llm.generate([prompt], sampling_params)
                raw_answer = out[0].outputs[0].text.strip()
                if "<think>" in raw_answer and "</think>" in raw_answer:
                    raw_answer = raw_answer.split("</think>")[-1].strip()

            pred_answer = extract_answer_from_box(raw_answer)

            final_score = None
            if pred_answer == gt_answer:
                final_score = 1
            else:
                tmp_prompt = (
                    CORRECTNESS_CHECK_PROMPT
                    .replace("<problem>", problem)
                    .replace("<correct>", gt_answer)
                    .replace("<incorrect>", raw_answer.strip())
                )
                for _attempt in range(3):
                    resp = get_completion(
                        [{"role": "user", "content": tmp_prompt}],
                        model=args.judge_model,
                        max_tokens=args.max_tokens_judge,
                        client=client,
                    )
                    if resp is None:
                        continue
                    label = resp.choices[0].message.content
                    if "yes" in label.lower():
                        final_score = 0
                        break
                    elif "no" in label.lower():
                        final_score = -1
                        break

            if final_score is None:
                continue

            result = {
                "idx": idx,
                "problem": problem,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "raw_answer": raw_answer.strip(),
                "correctness": final_score,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
