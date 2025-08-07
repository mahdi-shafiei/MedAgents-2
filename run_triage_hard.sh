#!/bin/bash
set -e

## Triage Ablation Study Runner
DATASETS=("medbullets" "medexqa" "medmcqa" "medqa" "medxpertqa-r" "medxpertqa-u" "mmlu" "mmlu-pro" "pubmedqa")
SPLIT="test_hard"
MODEL="o3-mini"
RUN_IDS=(0 1 2)

echo "=========================================="
echo "Running MedAgents-2 Triage Ablation Study"
echo "=========================================="
echo "Datasets: ${DATASETS[@]}"
echo "Split: $SPLIT"
echo "Model: $MODEL"
echo ""

## Group 1: Triage vs. No Triage
EXPERIMENT_NAME="ebagents"

declare -A configs
configs["new"]="triage.forced_level=hard search.search_mode=vector"

for RUN_ID in "${RUN_IDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do 
        for ablation in "${!configs[@]}"; do
            echo "Running $ablation ablation..."
            python run_experiments.py \
                execution.dataset.name=$DATASET \
                execution.dataset.split=$SPLIT \
                execution.model.name=$MODEL \
                execution.experiments.run_id=$RUN_ID \
                execution.experiment_name=${EXPERIMENT_NAME}/$ablation \
                ${configs[$ablation]}
            echo "Completed $ablation ablation."
            echo ""
        done
        echo "=========================================="
        echo "Triage ablation study completed!"
        echo "Results saved in: output/${DATASET}/"
        echo "=========================================="
        echo ""
    done
done
