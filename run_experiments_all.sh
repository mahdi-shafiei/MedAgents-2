#!/bin/bash
#SBATCH --job-name=medagents_experiments
#SBATCH --output=logs/medagents_experiments.log
#SBATCH --error=logs/medagents_experiments.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=scavenge

LOGS_DIR=logs
DATA_DIR=data

#for dataset in medqa medmcqa pubmedqa medbullets mmlu-pro mmlu; do
for dataset in medxpertqa-u medxpertqa-r medexqa; do
    mkdir -p $LOGS_DIR/$dataset
    for model in gpt-4o-mini; do
        for split in test_hard; do #sample_1_hard, test_hard
            for difficulty in adaptive; do
                date_folder=$(date +"%Y%m%d")
                mkdir -p $LOGS_DIR/$dataset/$date_folder
                log_file=$LOGS_DIR/$dataset/$date_folder/${model}_${dataset}_${split}_${difficulty}.log
                error_file=$LOGS_DIR/$dataset/$date_folder/${model}_${dataset}_${split}_${difficulty}.err
                echo "Running $model on $split with difficulty $difficulty"
                python main.py \
                --model_name $model \
                --dataset_name $dataset \
                --dataset_dir $DATA_DIR \
                --split $split \
                --output_files_folder ./output/ \
                --gpu_ids 4 5 6 7 \
                --num_processes 4 \
                --retrieve_topk 20 \
                --rerank_topk 32 \
                --rewrite True\
                --adaptive_rag auto \
                --similarity_strategy reuse \
                --agent_memory True \
                --device cpu \
                --splice_length 500 \
                --temperature 0 \
                --top_p 0.95 \
                --max_tokens 16384 \
                --presence_penalty 0.0 \
                --frequency_penalty 0.0 \
                --max_retries 5 \
                --query_similarity_threshold 0.85 > $log_file 2> $error_file
            done
        done
    done
done