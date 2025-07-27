#!/bin/bash
set -e

## MedRAG Experiment Runner
# Configuration - Available datasets: medbullets, medexqa, medmcqa, medqa, medxpertqa-r, medxpertqa-u, mmlu, mmlu-pro, pubmedqa
DATASETS=("medbullets" "medexqa" "medmcqa" "medqa" "medxpertqa-r" "medxpertqa-u" "mmlu" "mmlu-pro" "pubmedqa")
SPLIT="test_hard"
MODEL="gpt-4o-mini"
RUN_IDS=(0 1 2)

echo "=========================================="
echo "Running MedRAG Experiment"
echo "=========================================="
echo "Split: $SPLIT"
echo "Model: $MODEL"
echo ""

## MedRAG Configuration: 1 agent, 1 round, with search
EXPERIMENT_NAME="medrag"

declare -A configs
configs["1_agent_1_round_with_search"]="triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=1 triage.easy.search_mode=required search.rewrite=false search.review=false search.search_mode=vector orchestrate.discussion_mode=group_chat_voting_only"

for RUN_ID in "${RUN_IDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for ablation in "${!configs[@]}"; do
            echo "Dataset: $DATASET"
            echo "Running $ablation configuration..."
            python run_experiments.py \
                execution.dataset.name=$DATASET \
                execution.dataset.split=$SPLIT \
                execution.model.name=$MODEL \
                execution.experiments.run_id=$RUN_ID \
                execution.experiment_name=${EXPERIMENT_NAME}_$ablation \
                ${configs[$ablation]}
            echo "Completed $ablation configuration for $DATASET."
            echo ""
        done
    done
done

echo "=========================================="
echo "MedRAG experiment completed!"
echo "Results saved in: output/[DATASET]/"
echo "=========================================="