python -u run_azure_multi.py \
--model_name gpt4omini \
--dataset_name MedQA \
--dataset_dir ./datasets/MedQA/ \
--start_pos 0 \
--end_pos -1 \
--output_files_folder ./outputs/MedQA/50_sampled_hard_medqa/ \
--max_attempt_vote 3 \
--method base_direct \
--max_workers 30 \