#!/bin/bash

# MedAgents-2 Basic Experiment Runner
# This script runs basic experiments using the new hierarchical output structure

set -e

echo "=========================================="
echo "Running MedAgents-2 Basic Experiment"
echo "=========================================="

# Configuration - Available datasets: afrimedqa, medbullets, medexqa, medmcqa, medqa, medqa_5options, medxpertqa-r, medxpertqa-u, mmlu, mmlu-pro, pubmedqa
DATASETS=("medqa" "medbullets" "medexqa" "medmcqa" "medxpertqa-r" "medxpertqa-u" "mmlu" "mmlu-pro" "pubmedqa")
RUN_IDS=(0 1 2)
SPLIT="test_hard"
MODEL="gpt-4o-mini"
EXPERIMENT_NAME="medagents"

# Loop over all datasets and run IDs
for RUN_ID in "${RUN_IDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "Dataset: $DATASET"
        echo "Split: $SPLIT"
        echo "Model: $MODEL"
        echo "Run ID: $RUN_ID"
        echo "Experiment: $EXPERIMENT_NAME"
        echo ""

        # Run baseline experiment
        echo "Running baseline experiment for $DATASET (run $RUN_ID)..."
        python run_experiments.py \
            execution.dataset.name=$DATASET \
            execution.dataset.split=$SPLIT \
            execution.model.name=$MODEL \
            execution.experiments.run_id=$RUN_ID \
            execution.experiment_name=medagents/$EXPERIMENT_NAME \

        echo "Completed experiment for $DATASET (run $RUN_ID)"
        echo "Results saved in: output/$DATASET/$EXPERIMENT_NAME/run_$RUN_ID/$MODEL/"
        echo ""
    done
done

echo "=========================================="
echo "All basic experiments completed!"
echo "==========================================" 