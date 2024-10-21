python -u run_azure_multi.py \
--model_name gpt4o \
--dataset_name MedQA \
--dataset_dir ./datasets/MedQA/ \
--start_pos 0 \
--end_pos -1 \
--output_files_folder ./outputs/MedQA/domain_test/threshold_abmsall_ \
--max_attempt_vote 3 \
--method syn_verif \
--max_workers 10 \