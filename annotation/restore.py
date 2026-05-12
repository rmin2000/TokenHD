"""
Step 3 of the data pipeline: restore annotated text spans to exact substrings.

Critic models sometimes output slightly paraphrased versions of the erroneous
spans rather than verbatim copies. This script calls an LLM to locate the
best-matching exact substring in the original response for each annotated span.

Required environment variable:
    OPENAI_API_KEY

Input:  verbal_labeler_<critic>/chunk_<n>.jsonl
Output: verbal_labeler_<critic>/restored/chunk_<n>.jsonl
"""

import os
import json
import re
import argparse

from openai import OpenAI

from utils import load_jsonl, get_completion
from prompts.restore import RESTORE_PROMPT

parser = argparse.ArgumentParser(description="Restore annotated spans to exact substrings.")
parser.add_argument("--chunk", type=int, default=1)
parser.add_argument("--source_model_name", type=str, default="gpt-4o-mini")
parser.add_argument("--label_model_name", type=str, default="gpt-4o-mini")
parser.add_argument("--folder_name", type=str, required=True)
parser.add_argument("--restore_model", type=str, default="gpt-4o-mini",
                    help="Model used for text span restoration.")
parser.add_argument("--data_dir", type=str, default="data")
args = parser.parse_args()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def parse_result_tags(text):
    pattern = r"<result\s*(\d+)>(.*?)</result\s*\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match[1].strip() for match in matches]


ori_data_file = (
    f"{args.data_dir}/{args.source_model_name}/{args.folder_name}/"
    f"verbal_labeler_{args.label_model_name.split('/')[-1]}/chunk_{args.chunk}.jsonl"
)
save_data_file = (
    f"{args.data_dir}/{args.source_model_name}/{args.folder_name}/"
    f"verbal_labeler_{args.label_model_name.split('/')[-1]}/restored/chunk_{args.chunk}.jsonl"
)

data = load_jsonl(ori_data_file)
print(f"Loaded {len(data)} items from {ori_data_file}.")

if os.path.exists(save_data_file):
    start_index = len(load_jsonl(save_data_file))
    writing_var = "a"
else:
    os.makedirs(os.path.dirname(save_data_file), exist_ok=True)
    start_index = 0
    writing_var = "w"

with open(save_data_file, writing_var, encoding="utf-8") as f:
    for idx, item in enumerate(data):
        if idx < start_index:
            continue

        print(f"Processing {idx + 1}/{len(data)} | Chunk {args.chunk}")

        raw_answer = item["raw_answer"]
        label_index_list_raw = item["label_index_list"]

        additional_info = {k: v for k, v in item.items() if k != "label_index_list"}

        if all((isinstance(s, list) and len(s) == 0) or s == "No errors!" for s in label_index_list_raw):
            result = {"label_index_list": label_index_list_raw}
            result.update(additional_info)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            continue

        label_index_list = []
        label_index_part = []
        no_error_count = 0

        for tmp_label_list in label_index_list_raw:
            if "No errors" not in tmp_label_list:
                label_index_part.append(len(tmp_label_list))
                label_index_list.extend(tmp_label_list)
            else:
                no_error_count += 1

        error_list_index = [i for i, x in enumerate(label_index_list) if x not in raw_answer]
        error_list_all = [x for x in label_index_list if x not in raw_answer]

        for _iter in range(4):
            if len(error_list_all) == 0:
                break

            error_list = [
                f"<extract{j}>" + x + f"</extract{j}>"
                for j, x in enumerate(error_list_all)
                if "No errors" not in x
            ]
            user_query = RESTORE_PROMPT.format(
                original_text=raw_answer,
                extracted_text="\n".join(error_list).strip(),
            )

            resp = get_completion(
                [{"role": "user", "content": user_query}],
                model=args.restore_model,
                max_tokens=8096,
                client=client,
            )

            if resp is None:
                print("API response is None, stopping restore.")
                break

            auto_label = resp.choices[0].message.content
            new_verbal_label = parse_result_tags(auto_label)
            error_list_all_new = []

            for tmp_idx, content in enumerate(new_verbal_label):
                if tmp_idx >= len(error_list_index):
                    break
                if content not in raw_answer:
                    if "NO_MATCH_FOUND" not in content:
                        error_list_all_new.append(content)
                    else:
                        error_list_all_new.append(error_list_all[tmp_idx])
                else:
                    label_index_list[error_list_index[tmp_idx]] = content

            error_list_all = error_list_all_new
            error_list_index = [i for i, x in enumerate(label_index_list) if x not in raw_answer]
            print(f"Match count: {len(label_index_list) - len(error_list_all)} / {len(label_index_list)}")

        label_index_list_new = []
        part_index = 0
        for part_num in label_index_part:
            label_index_list_new.append(label_index_list[part_index : part_index + part_num])
            part_index += part_num
        label_index_list_new.extend(["No errors!"] * no_error_count)

        result = {"label_index_list": label_index_list_new}
        result.update(additional_info)
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
