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
RUN_ID=0

# Ablation over retrieve_topk and rerank_topk
for retrieve_topk in 5 10 20 40 80; do
    rerank_topk=$(python -c "print(max(1, int(round($retrieve_topk * 0.4))))")
    for dataset in medqa; do
        mkdir -p $LOGS_DIR/$dataset
        for model in gpt-4o-mini; do
            for split in test_hard; do #sample_1_hard, test_hard
                for difficulty in adaptive; do
                    mkdir -p $LOGS_DIR/$dataset/run_$RUN_ID
                    log_file=$LOGS_DIR/$dataset/run_${RUN_ID}/${model}_${dataset}_${split}_${difficulty}_retrieve${retrieve_topk}_rerank${rerank_topk}.log
                    error_file=$LOGS_DIR/$dataset/run_${RUN_ID}/${model}_${dataset}_${split}_${difficulty}_retrieve${retrieve_topk}_rerank${rerank_topk}.err
                    echo "Running $model on $split with difficulty $difficulty, retrieve_topk $retrieve_topk, rerank_topk $rerank_topk"
                    python main.py \
                    --model_name $model \
                    --dataset_name $dataset \
                    --dataset_dir $DATA_DIR \
                    --split $split \
                    --run_id $RUN_ID \
                    --output_files_folder ./output/ \
                    --gpu_ids 4 5 6 7 \
                    --num_processes 4 \
                    --retrieve_topk $retrieve_topk \
                    --rerank_topk $rerank_topk \
                    --rewrite True \
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
                    --gather_evidence few_shot \
                    --query_similarity_threshold 0.85 > $log_file 2> $error_file
                done
            done
        done
    done
done