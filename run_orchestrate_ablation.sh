#!/bin/bash
set -e

## Orchestrate Ablation Study Runner
DATASET="medqa"
SPLIT="test_hard"
MODEL="gpt-4o-mini"
RUN_IDS=(0)

echo "=========================================="
echo "Running MedAgents-2 Orchestrate Ablation Study"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Model: $MODEL"
echo ""

## Core Discussion Mode Ablation Study
EXPERIMENT_NAME="discussion_mode_ablation"

declare -A configs
configs["independent"]="orchestrate.discussion_mode=independent triage.disable_triage=true triage.forced_level=custom triage.forced_level_custom.num_experts=1 triage.forced_level_custom.max_rounds=3 search.search_mode=web"
configs["group_chat_with_orchestrator"]="orchestrate.discussion_mode=group_chat_with_orchestrator search.search_mode=web"
configs["group_chat_voting_only"]="orchestrate.discussion_mode=group_chat_voting_only search.search_mode=web"
configs["one_on_one_sync"]="orchestrate.discussion_mode=one_on_one_sync search.search_mode=web"

for RUN_ID in "${RUN_IDS[@]}"; do
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
done

echo "=========================================="
echo "Orchestrate ablation study completed!"
echo "Results saved in: output/$DATASET/"
echo "==========================================" 