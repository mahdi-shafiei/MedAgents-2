import json
import os
import random
import numpy as np

def save_as_json(json_list, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(json_list, outfile, indent=4)

def save_as_jsonl(json_list, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for json_obj in json_list:
            json.dump(json_obj, outfile)
            outfile.write('\n')

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    return data

def deduplicate_data(data):
    seen = set()
    deduplicated_data = []
    for item in data:
        idx = item['realidx']
        if idx not in seen:
            deduplicated_data.append(item)
            seen.add(idx)
    return deduplicated_data

def calculate_cost_from_token_usage(data, model):
    total_cost = 0
    for item in data:
        if model == 'gpt-4o-mini':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 0.15 / 1000000 + item['token_usage']['all']['completion_tokens'] * 0.6 / 1000000
        elif model == 'gpt-4o':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 2.5 / 1000000 + item['token_usage']['all']['completion_tokens'] * 10 / 1000000
        elif model == 'o3-mini' or model == 'o1-mini':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 1.1 / 1000000 + item['token_usage']['all']['completion_tokens'] * 4.4 / 1000000
        elif model == 'claude-3-5-sonnet':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 3.0 / 1000000 + item['token_usage']['all']['completion_tokens'] * 15.0 / 1000000
        elif model == 'claude-3-5-haiku':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 0.8 / 1000000 + item['token_usage']['all']['completion_tokens'] * 4.0 / 1000000
        elif model == 'qwq':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 1.2 / 1000000 + item['token_usage']['all']['completion_tokens'] * 1.2 / 1000000
        elif model == 'qwen2.5':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 0.3 / 1000000 + item['token_usage']['all']['completion_tokens'] * 0.3 / 1000000
        elif model == 'r1':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 2.5 / 1000000 + item['token_usage']['all']['completion_tokens'] * 7 / 1000000
        elif model == 'v3':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 1.25 / 1000000 + item['token_usage']['all']['completion_tokens'] * 1.25 / 1000000
        elif model == 'llama3.3':
            total_cost += item['token_usage']['all']['prompt_tokens'] * 0.88 / 1000000 + item['token_usage']['all']['completion_tokens'] * 0.88 / 1000000
    return total_cost / len(data)

def calculate_time_from_data(data):
    total_time = 0
    for item in data:
        total_time += item['time_taken']
    return total_time / len(data)

def get_score_from_file(file_path, model_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dedup = deduplicate_data(data)
    answers = []
    for item in dedup:
        # Some datasets may use different answer keys, adapt as needed
        gt = item.get('answer_idx') or item.get('answer')
        pred = item['answer_by_turns'][-1]['answer']
        answers.append(gt == pred)
    acc = np.mean(answers) * 100
    avg_cost = calculate_cost_from_token_usage(dedup, model_name)
    avg_time = calculate_time_from_data(dedup)
    return acc, len(dedup), os.path.basename(file_path), avg_cost, avg_time
