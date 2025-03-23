import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm.auto import tqdm
import torch
from constants import MEDICAL_SPECIALTIES_GPT_SELECTED
from dotenv import load_dotenv
from retriever import MedCPTRetriever
from agent import TriageUnit, SearchUnit, ModerationUnit, DiscussionUnit

load_dotenv()

FORMAT_INST = "Reply EXACTLY with the following JSON format.\n{format}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"

def save_results(results, existing_output_file):
    results = sorted(results, key=lambda x: x['realidx'])
    with open(existing_output_file, 'w') as f:
        json.dump(results, f, indent=4)

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def process_query(problem, args, process_idx):
    # Set device for this process based on its index
    device = f"cuda:{process_idx % len(args.gpu_ids)}" if torch.cuda.is_available() and args.gpu_ids else args.device
    
    retriever = MedCPTRetriever(device)
    triage_unit = TriageUnit(args)
    expert_list = triage_unit.run(problem['question'], problem['options'], MEDICAL_SPECIALTIES_GPT_SELECTED, 5)
    search_unit = SearchUnit(args, retriever, device)
    moderation_unit = ModerationUnit(args)
    discussion_unit = DiscussionUnit(args, expert_list, search_unit, moderation_unit)
    results = discussion_unit.run(problem['question'], problem['options'], args.llm_debate_max_round)
    problem['answer_by_turns'] = results
    return problem

def parse_args():
    parser = argparse.ArgumentParser(description='Medical Agent System Arguments')
    parser.add_argument('--model_name', default='gpt-4o-mini',
                        help='Model name')
    parser.add_argument('--dataset_name', type=str, default='medqa',
                        help='Dataset name')
    parser.add_argument('--dataset_dir', type=str, default='./data',
                        help='Dataset directory')
    parser.add_argument('--split', type=str, default='test_hard',
                        help='Split name')
    parser.add_argument('--output_files_folder', default='./output/',
                        help='Output files folder')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of processes')
    parser.add_argument('--allowed_sources', nargs='+', default=['cpg', 'statpearls', 'recop', 'textbooks'],
                        help='List of allowed source types')
    parser.add_argument('--llm_debate_max_round', type=int, default=1,
                        help='Maximum debate rounds')
    parser.add_argument('--retrieve_topk', type=int, default=100,
                        help='Top k documents to retrieve')
    parser.add_argument('--rerank_topk', type=int, default=25,
                        help='Top k documents to rerank')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[4, 5, 6, 7],
                        help='GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Temperature for LLM generation')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top p for LLM generation')
    parser.add_argument('--max_tokens', type=int, default=16384,
                        help='Maximum tokens for LLM generation')
    parser.add_argument('--presence_penalty', type=float, default=0.0,
                        help='Presence penalty for LLM generation')
    parser.add_argument('--frequency_penalty', type=float, default=0.0,
                        help='Frequency penalty for LLM generation')
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum retries for LLM generation')
    parser.add_argument('--rewrite', type=str, choices=['True', 'False', 'Both',], default='False',
                        help='Whether to use rewritten query, original query, or both')
    parser.add_argument('--review', type=str, choices=['True', 'False'], default='False',
                        help='Whether to review')
    parser.add_argument('--adaptive_rag', type=str, choices=['True', 'False'], default='False',
                        help='Whether to use adaptive rag during the debate')
    parser.add_argument('--naive_rag', type=str, choices=['True', 'False'], default='False',
                        help='Whether to use naive RAG at the beginning of the process')
    parser.add_argument('--decomposed_rag', type=str, choices=['True', 'False'], default='False',
                        help='Whether to use decomposed RAG at the beginning of the process')
    parser.add_argument('--query_similarity_threshold', type=float, default=0.85,
                        help='Similarity threshold for detecting similar queries in decomposed RAG')
    parser.add_argument('--agent_memory', type=str, nargs='+', default=['decompose_query', 'decompose_answer', 'debate'],
                        help='List of stages to save (e.g., decompose, triage, debate)')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Default device to use (cuda or cpu) when not using per-process GPU allocation')


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    os.makedirs(args.output_files_folder, exist_ok=True)
    subfolder = os.path.join(args.output_files_folder, args.dataset_name)
    os.makedirs(subfolder, exist_ok=True)
    existing_output_file = os.path.join(args.output_files_folder, args.dataset_name, f"{args.model_name}-{args.dataset_name}-{args.split}-rounds-{args.llm_debate_max_round}-retrieve-{args.retrieve_topk}-rerank-{args.rerank_topk}-rewrite-{args.rewrite}-review-{args.review}-adaptive-{args.adaptive_rag}-naive-{args.naive_rag}-decomposed-{args.decomposed_rag}-agent-memory-{args.agent_memory}.json")
    
    if os.path.exists(existing_output_file):
        print(f"Existing output file found: {existing_output_file}")
        with open(existing_output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from existing file.")
    else:
        results = []

    problems = load_jsonl(os.path.join(args.dataset_dir, args.dataset_name, f"{args.split}.jsonl"))
    for idx, problem in enumerate(problems):
        if 'realidx' not in problem:
            problem['realidx'] = idx

    processed_realidx = {result.get('realidx', None) for result in results}
    problems_to_process = [problem for problem in problems if problem['realidx'] not in processed_realidx]

    print(f"Processing {len(problems_to_process)} problems out of {len(problems)} total problems.")
    print(f"Using GPUs: {args.gpu_ids}")

    
    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        future_to_index = {
            executor.submit(process_query, problem, args, idx): idx
            for idx, problem in enumerate(problems_to_process)
        }
        for future in tqdm(as_completed(future_to_index),
                           total=len(future_to_index),
                           desc="Processing queries",
                           unit="problem"):
            result = future.result()
            results.append(result)
            save_results(results, existing_output_file)
    print(f"Saved {len(results)} results to {existing_output_file}")
    save_results(results, existing_output_file)
