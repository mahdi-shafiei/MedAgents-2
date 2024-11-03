from data_utils import MyDataset
from api_utils_azure import api_handler
from string import punctuation
import argparse
import tqdm
import json
from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed 

def process_sample(idx, raw_sample, realqid, handler, args, dataobj):
    question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'
    if args.dataset_name in ['MedQA', 'MedMCQA'] or 'MMLU' in args.dataset_name:
        options = raw_sample['options']
        gold_answer = raw_sample['answer_idx']
        return fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj)
    elif args.dataset_name == 'PubMedQA':
        question = raw_sample['context'] + ' ' + question
        options = raw_sample['options']
        gold_answer = raw_sample['answer_idx']
        return fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj)
    elif args.dataset_name in ['MedicationQA']:
        options = ''
        gold_answer = raw_sample['answer_idx']
        return fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='chatgpt')
    parser.add_argument('--dataset_name', default='MedQA')
    parser.add_argument('--dataset_dir', default='./datasets/MedQA/')
    parser.add_argument('--start_pos', type=int, default=21)
    parser.add_argument('--end_pos', type=int, default=50)
    parser.add_argument('--output_files_folder', default='./outputs/MedQA')
    parser.add_argument('--max_workers', type=int, default=40)
    

    parser.add_argument('--method', type=str, default='syn_verif', choices=['syn_verif', 'syn_only', 'anal_only', 'base_direct', 'base_cot'])
    parser.add_argument('--max_attempt_vote', type=int, default=3)
    args = parser.parse_args()

    print(args)

    ### get handler
    if args.model_name in ['instructgpt', 'newinstructgpt', 'chatgpt', 'gpt4', 'gpt4o','gpt4omini', 'llama3.1-70b', 'qwen2.5-72b', 'deepseek-V2-67b']: # select the model
        handler = api_handler(args.model_name)
    else:
        raise ValueError

    ### get dataobj
    dataobj = MyDataset('test', args, traindata_obj=None)

    ### set test range
    end_pos = len(dataobj) if args.end_pos == -1 else args.end_pos
    test_range = range(args.start_pos, end_pos)  # closed interval

    ### set output_file_name
    exact_output_file = f"{args.output_files_folder}{args.model_name}-{args.method}"
    #print(exact_output_file)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        
        # Submit tasks to the executor
        for idx in tqdm.tqdm(test_range, desc=f"{args.start_pos} ~ {end_pos}"):
            raw_sample = dataobj.get_by_idx(idx)
            realqid = idx
            # Submit each fully_decode task to the thread pool
            futures.append(executor.submit(process_sample, idx, raw_sample, realqid, handler, args, dataobj))
        
        results_dict = {}

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            try:
                data_info = future.result()
                idx = data_info['idx']  # Ensure 'idx' is available in the result
                results_dict[idx] = data_info  # Store result with its index
            except Exception as e:
                print(f"Error processing sample: {e}")

        # Write results in the original order of idx
        with open(exact_output_file, 'a') as f:
            for idx in sorted(results_dict.keys()):
                record = json.dumps(results_dict[idx])
                f.write(record + '\n')