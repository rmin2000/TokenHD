import json
import re
import glob
import numpy as np
import time
from openai import OpenAI


def get_completion(
    messages,
    model="gpt-4o-mini",
    max_tokens=2048,
    temperature=0.7,
    top_p=0.95,
    max_retries=3,
    reasoning_effort=None,
    client=None,
):
    for attempt in range(max_retries):
        try:
            if model in ["o1-mini", "o3-mini", "o4-mini", "gpt-5"] and reasoning_effort is None:
                params = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                }
                completion = client.chat.completions.create(**params)
            elif model in ["o3", "o4-mini", "gpt-5"] and reasoning_effort is not None:
                completion = client.responses.create(
                    model=model,
                    reasoning={"effort": reasoning_effort},
                    input=messages,
                )
                completion = completion.output_text
            else:
                params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                completion = client.chat.completions.create(**params)
            return completion
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"All {max_retries} attempts failed: {e}")
                return None
            time.sleep(2 ** attempt)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def parse_error_tags(text):
    pattern = r"<error\s*(\d+)>(.*?)</error\s*\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match[1].strip() for match in matches]


def restore_text(prompt_template, auto_label_list, raw_answer, client, model="gpt-4o-mini", max_tokens=8096):
    def parse_extract_tags(text):
        pattern = r"<result\s*(\d+)>(.*?)</result\s*\1>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match[1].strip() for match in matches]

    label_index_list = []
    label_index_part = []
    no_error_count = 0

    for tmp_label_list in auto_label_list:
        if "No errors" not in tmp_label_list:
            label_index_part.append(len(tmp_label_list))
            label_index_list.extend(tmp_label_list)
        else:
            no_error_count += 1

    error_list_index = [i for i, x in enumerate(label_index_list) if x not in raw_answer]
    error_list_all = [x for x in label_index_list if x not in raw_answer]

    for _ in range(4):
        if len(error_list_all) == 0:
            break
        error_list = [
            f"<extract{j}>" + x + f"</extract{j}>"
            for j, x in enumerate(error_list_all)
            if "No errors" not in x
        ]
        error_input = "\n".join(error_list).strip()

        user_query = prompt_template.format(
            original_text=raw_answer,
            extracted_text=error_input,
        )

        API_RESPONSE = get_completion(
            [{"role": "user", "content": user_query}],
            model=model,
            max_tokens=max_tokens,
            client=client,
        )

        if API_RESPONSE is not None:
            error_list_all_new = []
            auto_label = API_RESPONSE.choices[0].message.content
            new_verbal_label = parse_extract_tags(auto_label)

            for tmp_idx, content in enumerate(new_verbal_label):
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
        else:
            print("API response is None, skipping remaining restore iterations.")
            break

    label_index_list_new = []
    part_index = 0
    for part_num in label_index_part:
        label_index_list_new.append(label_index_list[part_index : part_index + part_num])
        part_index += part_num
    label_index_list_new.extend(["No errors!"] * no_error_count)
    return label_index_list_new


def extract_answer_from_box(latex_str):
    if latex_str is None:
        return None
    last_box_pos = latex_str.rfind("\\box{")
    last_boxed_pos = latex_str.rfind("\\boxed{")

    if last_box_pos == -1 and last_boxed_pos == -1:
        return None

    if last_box_pos > last_boxed_pos:
        start_tag = "\\box{"
        start_pos = last_box_pos
    else:
        start_tag = "\\boxed{"
        start_pos = last_boxed_pos

    search_start = start_pos + len(start_tag)
    brace_level = 1

    for i in range(search_start, len(latex_str)):
        char = latex_str[i]
        if char == "{":
            brace_level += 1
        elif char == "}":
            brace_level -= 1
        if brace_level == 0:
            return latex_str[search_start:i]
    return None


def hard_f1(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return precision, recall, f1


def token_ensemble(raw_answer, label_list, critic_num, tokenizer, correctness=1, return_suffix=False):
    encoding = tokenizer(raw_answer, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding["input_ids"]
    token_weights = np.array([0.0] * len(token_ids))

    if critic_num == 0:
        if return_suffix:
            return token_ids, token_weights, -1
        else:
            return token_ids, token_weights

    offset_mapping = encoding["offset_mapping"]
    ensemble_char_count = np.array([0.0] * len(raw_answer))

    for tmp_label in label_list:
        assert tmp_label in raw_answer
        start_index = raw_answer.find(tmp_label)
        end_index = start_index + len(tmp_label)
        ensemble_char_count[start_index:end_index] += 1

    ensemble_char_count = np.clip(ensemble_char_count / critic_num, 0, 1)

    for i in range(len(token_ids)):
        start_char, end_char = offset_mapping[i]
        token_weights[i] = np.sum(ensemble_char_count[start_char:end_char]) / (end_char - start_char)

    best_start = -1
    best_end = -1
    separators = [", ", ".\n", ". "]
    box_pos = raw_answer.find("\\box")
    if box_pos == -1:
        box_pos = len(raw_answer)

    for separator in separators:
        search_text = raw_answer[:box_pos]
        last_sep_pos = search_text.rfind(separator)
        if last_sep_pos != -1:
            start_pos = last_sep_pos + len(separator)
            if start_pos > best_start:
                best_start = start_pos
                best_end = len(raw_answer)

    if correctness == -1 and best_start != -1:
        ensemble_char_count[best_start:best_end] = 1.0
        for i in range(len(token_ids)):
            start_char, end_char = offset_mapping[i]
            token_weights[i] = np.sum(ensemble_char_count[start_char:end_char]) / (end_char - start_char)

    if return_suffix:
        suffix_start_token_idx = -1
        if best_start != -1:
            for i, (start_char, end_char) in enumerate(offset_mapping):
                if start_char <= best_start < end_char:
                    suffix_start_token_idx = i
                    break
            if suffix_start_token_idx == -1:
                for i, (start_char, end_char) in enumerate(offset_mapping):
                    if start_char >= best_start:
                        suffix_start_token_idx = i
                        break
        return token_ids, token_weights, suffix_start_token_idx
    else:
        return token_ids, token_weights


def token_ensemble_weighted(raw_answer, label_list, weight_list, tokenizer):
    encoding = tokenizer(raw_answer, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding["input_ids"]
    token_weights = np.array([0.0] * len(token_ids))

    if not weight_list:
        return token_ids, token_weights

    offset_mapping = encoding["offset_mapping"]
    ensemble_char_count = np.array([0.0] * len(raw_answer))

    for weight, tmp_label in zip(weight_list, label_list):
        assert tmp_label in raw_answer
        start_index = raw_answer.find(tmp_label)
        end_index = start_index + len(tmp_label)
        ensemble_char_count[start_index:end_index] += weight

    ensemble_char_count = np.clip(ensemble_char_count, 0, 1)

    for i in range(len(token_ids)):
        start_char, end_char = offset_mapping[i]
        token_weights[i] = np.sum(ensemble_char_count[start_char:end_char]) / (end_char - start_char)

    return token_ids, token_weights


def data_preprocess(label_model_list, policy_model, folder_name, save_folder_name):
    label_list_all = []
    for model_name in label_model_list:
        temp_file_data = []
        chunk_file_name_list = glob.glob(
            f"{save_folder_name}/{policy_model}/{folder_name}/verbal_labeler_{model_name}/restored/chunk_*.jsonl"
        )

        if len(chunk_file_name_list) == 0:
            print(f"No chunk files found for model {model_name} in folder {folder_name}.")
            continue

        chunk_files = []
        for file_name in chunk_file_name_list:
            m = re.search(r"chunk_(\d+)", file_name)
            if m:
                chunk_files.append((int(m.group(1)), file_name))

        if len(chunk_files) == 0:
            print(f"No valid chunk files found for model {model_name} in folder {folder_name}.")
            continue

        chunk_files.sort(key=lambda x: x[0])

        for _, file_path in chunk_files:
            for item in load_jsonl(file_path):
                temp_file_data.append(item)

        if "math" in folder_name:
            temp_file_data = [
                item
                for item in temp_file_data
                if not (extract_answer_from_box(item["raw_answer"]) is None and item["correctness"] == -1)
            ]

        print(f"{model_name}: {len(temp_file_data)}")
        label_list_all.append(temp_file_data)
    return label_list_all
