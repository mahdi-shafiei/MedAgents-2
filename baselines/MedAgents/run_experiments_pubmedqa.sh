#!/bin/bash
#SBATCH --job-name=medagents_experiments
#SBATCH --output=logs/medqa_experiments.log
#SBATCH --error=logs/medqa_experiments.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=scavenge

logs_dir=logs
mkdir -p $logs_dir
mkdir -p $logs_dir/pubmedqa
logs_dir=$logs_dir/pubmedqa

for model in gpt-4o; do
    for split in sampled_50; do
        for difficulty in adaptive; do
            log_file=$logs_dir/${model}_pubmedqa_${split}_${difficulty}.log
            error_file=$logs_dir/${model}_pubmedqa_${split}_${difficulty}.err
            echo "Running $model on $split with difficulty $difficulty"
            python main.py --dataset_name pubmedqa --dataset_dir ./data/pubmedqa/ --split $split --model $model --method syn_verif --output_files_folder ./output/ --num_processes 4 > $log_file 2> $error_file
        done
    done
done
