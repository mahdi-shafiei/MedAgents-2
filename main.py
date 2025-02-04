import os
from abc import ABC, abstractmethod
import argparse
import re
import json
import time
import utils
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI, OpenAI
from pymilvus import MilvusClient
from datetime import datetime
from tqdm.auto import tqdm
import torch
from constants import MEDICAL_SPECIALTIES_GPT_SELECTED
from dotenv import load_dotenv
from retriever import MedCPTRetriever
from agent import TriageUnit, SearchUnit, ModerationUnit, DiscussionUnit
load_dotenv()

FORMAT_INST = "Reply EXACTLY with the following JSON format.\n{format}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_query(question, choices, args):
    retriever = MedCPTRetriever(device)
    triage_unit = TriageUnit(args)
    expert_list = triage_unit.run(question, choices, MEDICAL_SPECIALTIES_GPT_SELECTED, 5)
    search_unit = SearchUnit(args, retriever, device)
    moderation_unit = ModerationUnit(args)
    discussion_unit = DiscussionUnit(args,expert_list, search_unit, moderation_unit)
    results = discussion_unit.run(question, choices, args.llm_debate_max_round)
    return results, search_unit.get_oom_count()

def parse_args():
    parser = argparse.ArgumentParser(description='Medical Agent System Arguments')
    parser.add_argument('--dataset_name', type=str, default='medqa',
                        help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='./data',
                        help='Dataset directory')
    parser.add_argument('--split', type=str, default='test_hard',
                        help='Split name')
    parser.add_argument('--allowed_sources', nargs='+', default=['cpg_2', 'statpearls_2', 'recop_2', 'textbook_2'],
                    help='List of allowed source types')
    parser.add_argument('--sample_size', type=int, default=1,
                        help='Sample size')
    parser.add_argument('--start_idx', type=int, default=19,
                        help='Starting index')
    parser.add_argument('--max_workers', type=int, default=30,
                        help='Maximum number of workers')
    parser.add_argument('--llm_debate_max_round', type=int, default=5,
                        help='Maximum debate rounds')
    parser.add_argument('--retrieve_topk', type=int, default=100,
                        help='Top k documents to retrieve')
    parser.add_argument('--rerank_topk', type=int, default=25,
                        help='Top k documents to rerank')
    parser.add_argument('--rewrite', action='store_true', default=False,
                        help='Whether to rewrite')
    parser.add_argument('--review', action='store_true', default=False,
                        help='Whether to review')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='GPU IDs to use')
    parser.add_argument('--voting', type=str, default='singular',
                        help='Voting type: singular, multi_ranked, multi_rated, or multi_points')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini',
                        help='LLM model to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Temperature for LLM generation')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top p for LLM generation')
    parser.add_argument('--max_tokens', type=int, default=32768,
                        help='Maximum tokens for LLM generation')
    parser.add_argument('--presence_penalty', type=float, default=0.0,
                        help='Presence penalty for LLM generation')
    parser.add_argument('--frequency_penalty', type=float, default=0.0,
                        help='Frequency penalty for LLM generation')
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum retries for LLM generation')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Include script name in the output directory
    output_dir = (
        f"./output/{script_name}/"
        f"results_llm_{llm_model}_rounds_{llm_debate_max_round}_retrieve_{retrieve_topk}_"
        f"rerank_{rerank_topk}_rewrite_{rewrite}_review_{review}_time_{current_time}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    oom_count_total = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(process_query, question, choices, task_number, gpu_ids): task_number
            for task_number, (question, choices) in enumerate(queries)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing queries"):
            idx = future_to_index[future]
            try:
                result, oom_count = future.result()  
                results[idx] = result
                oom_count_total += oom_count
            except Exception as e:
                print(f"Error: {str(e)}")
                results[idx] = f"Error: {str(e)}"  

    print(f"Total OOM errors encountered: {oom_count_total}")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    with open(os.path.join(output_dir, "results.json"),'w') as jsfile:
        json.dump(results, jsfile)
    for debate_round in range(llm_debate_max_round):
        count = 0
        for i in range(len(medqa_test)):
            if results[i][debate_round] == medqa_test[i]['answer_idx']:
                count += 1
        print(count/len(medqa_test))
