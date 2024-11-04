import os
import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
from multiprocessing import Pool
from utils import (
    Agent, Group, parse_hierarchy, parse_group_info, setup_model,
    load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query
)

from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='medqa')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--model', type=str, default='gpt-4o-mini')
parser.add_argument('--difficulty', type=str, default='adaptive')
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--num_processes', type=int, default=1)
args = parser.parse_args()

model, client = setup_model(args.model)
test_qa, examplers = load_data(args.dataset, args.split)

agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
random.shuffle(agent_emoji)

path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(path):
    os.makedirs(path)

if args.num_samples is None:
    args.num_samples = len(test_qa)

results_path = f'output/{args.model}_{args.dataset}_{args.split}_{args.difficulty}.json'
if os.path.exists(results_path):
    with open(results_path, 'r') as file:
        results = json.load(file)
else:
    results = []

# if test.jsonl doesn't have realidx, annotate with 
if 'realidx' not in test_qa[0]:
    test_qa = [{**s, 'realidx': idx} for idx, s in enumerate(test_qa)]

processed_idx = set([r['idx'] for r in results])
new_samples = [s for s in test_qa if s['realidx'] not in processed_idx]
if args.num_samples is not None:
    new_samples = new_samples[:min(args.num_samples, len(new_samples))]

def process_sample(sample):
    try:
        question, _ = create_question(sample, args.dataset)
        difficulty = determine_difficulty(question, args.difficulty, args.model)

        print(f"difficulty: {difficulty}")

        if difficulty == 'basic':
            final_decision = process_basic_query(question, examplers, args.model, args)
        elif difficulty == 'intermediate':
            final_decision = process_intermediate_query(question, examplers, args.model, args)
        elif difficulty == 'advanced':
            final_decision = process_advanced_query(question, args.model, args)

        return {
            'idx': sample['realidx'],
            'question': question,
            'label': sample['answer_idx'],
            'answer': sample['answer'],
            'options': sample['options'],
            'response': final_decision['majority'],
            'prediction': final_decision['answer'],
            'difficulty': difficulty
        }
    except Exception as e:
        print(f"[ERROR] Processing sample {sample['realidx']} failed: {e}")
        return None

if args.num_processes > 1:
    with Pool(args.num_processes) as p:
        for result in tqdm(p.imap(process_sample, new_samples), total=len(new_samples)):
            if result is not None:
                results.append(result)
else:
    for no, sample in enumerate(tqdm(new_samples)):
        result = process_sample(sample)
        if result is not None:
            results.append(result)

with open(results_path, 'w') as file:
    json.dump(results, file, indent=4)