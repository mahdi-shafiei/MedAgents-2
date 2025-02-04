import os
import json
from typing import Dict, Any, List
from tqdm import tqdm
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import argparse
from pydantic import BaseModel
import time

load_dotenv()

class AnswerResponse(BaseModel):
    answer_idx: str


def zero_shot(problem: Dict, client: Any, model: str = "o1-mini", retries: int = 3) -> Dict:
    question_text = problem.get('question', '')
    options = problem.get('options', {})
    options_text = ' '.join([f"({key}) {value}" for key, value in options.items()])

    answer_schema = {
        "name": "answer_response",
        "schema": {
            "type": "object",
            "properties": {
                "answer_idx": {"type": "string", "enum": list(options.keys())}
            },
            "required": ["answer_idx"],
            "additionalProperties": False
        },
        "strict": True
    }

    # Construct the prompt ensuring a structured JSON output that matches the provided schema.
    prompt = (
        "You are a knowledgeable medical assistant. Provide accurate answers to the medical question based on the given information. "
        "Return your answer as a JSON object that strictly follows the provided schema."
        f"\nQuestion:\n{question_text}\n\n"
        f"Options:\n{options_text}"
    )

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(retries):
        try:
            start_time = time.time()
            # Step 1: Call the designated model to get an initial answer.
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": answer_schema}
            )
            raw_response = completion.choices[0].message.content.strip()
            predicted_answer = AnswerResponse.parse_raw(raw_response).answer_idx.strip()
            usage = completion.usage
            total_prompt_tokens = usage.prompt_tokens
            total_completion_tokens = usage.completion_tokens
            end_time = time.time()
            time_elapsed = end_time - start_time

            problem['predicted_answer'] = predicted_answer
            problem['token_usage'] = {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            }
            problem['time_elapsed'] = time_elapsed
            return problem
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                return problem
            continue

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_results(results, existing_output_file):
    results = sorted(results, key=lambda x: x['realidx'])
    with open(existing_output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='o3-mini')
    parser.add_argument('--dataset_name', default='medqa')
    parser.add_argument('--dataset_dir', default='./data/medqa/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    parser.add_argument('--output_files_folder', default='./output/')
    parser.add_argument('--num_processes', type=int, default=4)

    args = parser.parse_args()
    
    client_old = AzureOpenAI(
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
    )
    
    if args.model_name == "o3-mini":
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_API_VERSION_2"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT_2"),
            api_key=os.getenv("AZURE_API_KEY_2"),
        )
    else:
        client = client_old

    os.makedirs(args.output_files_folder, exist_ok=True)
    subfolder = os.path.join(args.output_files_folder, args.dataset_name)
    os.makedirs(subfolder, exist_ok=True)
    existing_output_file = os.path.join(args.output_files_folder, args.dataset_name, f"{args.model_name}-{args.dataset_name}-{args.split}-zero_shot.json")
    
    if os.path.exists(existing_output_file):
        print(f"Existing output file found: {existing_output_file}")
        with open(existing_output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from existing file.")
    else:
        results = []

    problems = load_jsonl(os.path.join(args.dataset_dir, f"{args.split}.jsonl"))
    for idx, problem in enumerate(problems):
        if 'realidx' not in problem:
            problem['realidx'] = idx

    processed_realidx = {result.get('realidx', None) for result in results}
    problems_to_process = [problem for problem in problems if problem['realidx'] not in processed_realidx]

    print(f"Processing {len(problems_to_process)} problems out of {len(problems)} total problems.")

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = {executor.submit(zero_shot, problem, client, args.model_name): problem for problem in problems_to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing problems", unit="problem"):
            try:
                result = future.result()
                result.pop('cleanse_cot')
                result.pop('predicted_answer_base_direct')
                results.append(result)
                save_results(results, existing_output_file)
            except Exception as e:
                print(f"Error processing a problem: {e}")

    save_results(results, existing_output_file)
    print(f"Saved {len(results)} results to {existing_output_file}")