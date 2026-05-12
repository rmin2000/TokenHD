"""
Step 2 of the data pipeline: annotate hallucinations in LLM responses.

Each response is passed to a critic model (local or API-based) which identifies
erroneous text spans using structured error tags. Multiple rollouts per sample
are supported to reduce annotation variance.

Required environment variable:
    OPENAI_API_KEY  — required when using --use_openai_key

Output format (JSONL):
    {
        "token_label_question_idx": int,
        "label_index_list": list[list[str] | "No errors!"],
        ...  (all fields from the input sample are preserved)
    }
"""

import os
import json
import re
import argparse
import torch

from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils import load_jsonl, get_completion
from prompts.judge import math_process_error_prompt, code_process_error_prompt, sci_process_error_prompt

parser = argparse.ArgumentParser(description="Annotate hallucinations in LLM responses.")
parser.add_argument("--source_model_name", type=str, default="gpt-4o-mini",
                    help="Policy model that generated the responses.")
parser.add_argument("--label_model_name", type=str, required=True,
                    help="Critic model used for annotation.")
parser.add_argument("--use_openai_key", action="store_true",
                    help="Use OpenAI API instead of local vLLM.")
parser.add_argument("--pipeline_parallel_size", type=int, default=1)
parser.add_argument("--tensor_parallel_size", type=int, default=1)
parser.add_argument("--max_tokens", type=int, default=20000)
parser.add_argument("--rollout_num", type=int, default=1,
                    help="Number of annotation rollouts per sample.")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--chunk", type=int, default=1)
parser.add_argument("--tot_chunk", type=int, default=1)
parser.add_argument("--folder_name", type=str, required=True)
parser.add_argument("--reasoning_effort", type=str, default=None,
                    choices=["low", "medium", "high"])
parser.add_argument("--error_only", action="store_true",
                    help="Only annotate responses marked as incorrect.")
parser.add_argument("--data_dir", type=str, default="data",
                    help="Root directory for data files.")
args = parser.parse_args()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def parse_error_tags(text):
    pattern = r"<error\s*(\d+)>(.*?)</error\s*\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match[1].strip() for match in matches]


data_file = f"{args.data_dir}/{args.source_model_name}/{args.folder_name}/all.jsonl"
solution_data = load_jsonl(data_file)
print(f"Loaded {len(solution_data)} items from {data_file}.")

solution_data = [
    item for item in solution_data
    if not (item.get("pred_answer") is None and item.get("correctness") == -1)
]

assert args.chunk >= 1, "Chunk index must start from 1."
cur_chunk = args.chunk - 1
if cur_chunk == args.tot_chunk - 1:
    solution_data = solution_data[cur_chunk * len(solution_data) // args.tot_chunk :]
else:
    solution_data = solution_data[
        cur_chunk * len(solution_data) // args.tot_chunk :
        (cur_chunk + 1) * len(solution_data) // args.tot_chunk
    ]
print(f"Chunk {args.chunk}/{args.tot_chunk}: {len(solution_data)} items.")

error_index_list = [i for i, x in enumerate(solution_data) if x["correctness"] == -1]

if args.folder_name in ["gpqa", "olym_phy", "fin_qa"]:
    system_prompt = sci_process_error_prompt
elif "code" in args.folder_name:
    system_prompt = code_process_error_prompt
else:
    system_prompt = math_process_error_prompt

save_file = (
    f"{args.data_dir}/{args.source_model_name}/{args.folder_name}/"
    f"verbal_labeler_{args.label_model_name.split('/')[-1]}/chunk_{args.chunk}.jsonl"
)
os.makedirs(os.path.dirname(save_file), exist_ok=True)

if not args.use_openai_key:
    print("Using vLLM for local inference.")
    available_gpus = torch.cuda.device_count()
    assert available_gpus > 0, "No GPUs available."
    llm = LLM(
        model=args.label_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.label_model_name, trust_remote_code=True)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=0,
        n=args.rollout_num,
        skip_special_tokens=False,
        temperature=args.temperature,
        top_p=0.95,
    )

if os.path.exists(save_file):
    existing_index_list = [item["token_label_question_idx"] for item in load_jsonl(save_file)]
    writing_var = "a"
else:
    existing_index_list = []
    writing_var = "w"

with open(save_file, writing_var, encoding="utf-8") as f:
    for idx in range(len(solution_data)):
        print(f"Processing {idx + 1}/{len(solution_data)} | {args.label_model_name} | Chunk {args.chunk}")
        if idx in existing_index_list:
            continue
        if args.error_only and idx not in error_index_list:
            continue

        problem = solution_data[idx]["problem"]
        raw_answer = solution_data[idx]["raw_answer"]
        additional_info = dict(solution_data[idx])

        user_query = system_prompt.format(problem=problem, solution=raw_answer.strip())
        auto_label_list = []

        if args.use_openai_key:
            for _ in range(args.rollout_num):
                resp = get_completion(
                    [{"role": "user", "content": user_query}],
                    model=args.label_model_name,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    client=client,
                )
                if isinstance(resp, str):
                    auto_label_list.append(resp.strip())
                elif resp is not None:
                    auto_label_list.append(resp.choices[0].message.content.strip())
                else:
                    auto_label_list.append("")
        else:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_query}],
                tokenize=False,
                add_generation_prompt=True,
            )
            out = llm.generate([prompt], sampling_params)
            auto_label_list = [out[0].outputs[i].text.strip() for i in range(len(out[0].outputs))]

            for i in range(len(auto_label_list)):
                label = auto_label_list[i]
                if "</think>" in label:
                    label = label.split("</think>")[-1].strip()
                auto_label_list[i] = label

        label_index_list = []
        for auto_label in auto_label_list:
            if "No errors" in auto_label:
                label_index_list.append("No errors!")
            else:
                label_index_list.append([span.strip() for span in parse_error_tags(auto_label)])

        result = {
            "token_label_question_idx": idx,
            "label_index_list": label_index_list,
        }
        result.update(additional_info)

        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
