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
for dataset in medqa; do
    mkdir -p $LOGS_DIR/$dataset
    for model in gpt-4o-mini; do
        for split in test_hard; do
            for difficulty in adaptive; do
                log_file=$LOGS_DIR/$dataset/${model}_${dataset}_${split}_${difficulty}.log
                error_file=$LOGS_DIR/$dataset/${model}_${dataset}_${split}_${difficulty}.err
                echo "Running $model on $split with difficulty $difficulty"
                python main.py \
                --model_name $model \
                --dataset_name $dataset \
                --dataset_dir $DATA_DIR \
                --split $split \
                --output_files_folder ./output/ \
                --num_processes 8 \
                --llm_debate_max_round 5 \
                --retrieve_topk 100 \
                --rerank_topk 25 \
                --rewrite Both \
                --review False \
                --adaptive_rag False \
                --naive_rag True \
                --decomposed_rag False \
                --agent_memory False > $log_file 2> $error_file
            done
        done
    done
done