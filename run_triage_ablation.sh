#!/bin/bash
set -e

## Triage Ablation Study Runner
DATASET="medqa"
SPLIT="test_hard"
MODEL="gpt-4o-mini"
RUN_IDS=(0)

echo "=========================================="
echo "Running MedAgents-2 Triage Ablation Study"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Model: $MODEL"
echo ""

## Group 1: Triage vs. No Triage
EXPERIMENT_NAME="triage_enabled_vs_disabled"

declare -A configs
configs["triage_enabled"]=""
configs["triage_disabled_easy"]="triage.disable_triage=true triage.forced_level=easy search.search_mode=web"
configs["triage_disabled_medium"]="triage.disable_triage=true triage.forced_level=medium search.search_mode=web"
configs["triage_disabled_hard"]="triage.disable_triage=true triage.forced_level=hard search.search_mode=web"

for RUN_ID in "${RUN_IDS[@]}"; do
    for ablation in "${!configs[@]}"; do
        echo "Running $ablation ablation..."
        python run_experiments.py \
            execution.dataset.name=$DATASET \
            execution.dataset.split=$SPLIT \
            execution.model.name=$MODEL \
            execution.experiments.run_id=$RUN_ID \
            execution.experiment_name=${EXPERIMENT_NAME}_$ablation \
            ${configs[$ablation]}
        echo "Completed $ablation ablation."
        echo ""
    done
done

## Group 2: Agent Configuration with Triage Disabled
EXPERIMENT_NAME="agent_configuration"

declare -A configs
configs["1_agent_1_round_no_search"]="triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=1 triage.easy.search_mode=none search.search_mode=web"
configs["1_agent_1_round_with_search"]="triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=1 triage.easy.search_mode=required search.search_mode=web"
configs["3_agents_1_round_no_search"]="triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=3 triage.easy.max_rounds=1 triage.easy.search_mode=none search.search_mode=web"
configs["3_agents_1_round_with_search"]="triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=3 triage.easy.max_rounds=1 triage.easy.search_mode=required search.search_mode=web"
configs["3_agents_2_rounds_with_search"]="triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=3 triage.easy.max_rounds=2 triage.easy.search_mode=required search.search_mode=web"
configs["3_agents_3_rounds_with_search"]="triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=3 triage.easy.max_rounds=3 triage.easy.search_mode=required search.search_mode=web"

for RUN_ID in "${RUN_IDS[@]}"; do
    for ablation in "${!configs[@]}"; do
        echo "Running $ablation ablation..."
        python run_experiments.py \
            execution.dataset.name=$DATASET \
            execution.dataset.split=$SPLIT \
            execution.model.name=$MODEL \
            execution.experiments.run_id=$RUN_ID \
            execution.experiment_name=${EXPERIMENT_NAME}_$ablation \
            ${configs[$ablation]}
        echo "Completed $ablation ablation."
        echo ""
    done
done

## Group 3: Custom Configuration Experiments
EXPERIMENT_NAME="custom_configuration"

declare -A configs
configs["2_agents_3_rounds_with_search"]="triage.disable_triage=true triage.forced_level=custom triage.forced_level_custom.num_experts=2 triage.forced_level_custom.max_rounds=3 triage.forced_level_custom.search_mode=required search.search_mode=web"
configs["5_agents_1_round_with_search"]="triage.disable_triage=true triage.forced_level=custom triage.forced_level_custom.num_experts=5 triage.forced_level_custom.max_rounds=1 triage.forced_level_custom.search_mode=required search.search_mode=web"
configs["1_agent_3_rounds_with_search"]="triage.disable_triage=true triage.forced_level=custom triage.forced_level_custom.num_experts=1 triage.forced_level_custom.max_rounds=3 triage.forced_level_custom.search_mode=required search.search_mode=web"

for RUN_ID in "${RUN_IDS[@]}"; do
    for ablation in "${!configs[@]}"; do
        echo "Running $ablation ablation..."
        python run_experiments.py \
            execution.dataset.name=$DATASET \
            execution.dataset.split=$SPLIT \
            execution.model.name=$MODEL \
            execution.experiments.run_id=$RUN_ID \
            execution.experiment_name=${EXPERIMENT_NAME}_$ablation \
            ${configs[$ablation]}
        echo "Completed $ablation ablation."
        echo ""
    done
done

echo "=========================================="
echo "Triage ablation study completed!"
echo "Results saved in: output/$DATASET/"
echo "=========================================="