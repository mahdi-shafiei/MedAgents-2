#!/bin/bash
set -e

## Experiment 1: Search Features
DATASET="medqa"
SPLIT="test_hard"
MODEL="gpt-4o-mini"
RUN_IDS=(0)
EXPERIMENT_NAME="search_features"

declare -A configs
configs["baseline"]="search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=true search.review=true search.search_mode=both"
configs["no_query_rewrite"]="search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=false search.review=true search.search_mode=both"
configs["no_document_review"]="search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=true search.review=false search.search_mode=both"
configs["no_rewrite_no_review"]="search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=false search.review=false search.search_mode=both"

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

## Experiment 2: Search Modality
EXPERIMENT_NAME="search_modality"

declare -A configs
configs["web_only"]="search.search_mode=web"
configs["vector_only"]="search.search_mode=vector"
configs["both"]="search.search_mode=both"

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

## Experiment 3: Search History
EXPERIMENT_NAME="search_history"

declare -A configs
configs["individual"]="search.search_history=individual search.search_mode=both"
configs["shared"]="search.search_history=shared search.search_mode=both"

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

## Experiment 4: Search Source Depth
EXPERIMENT_NAME="search_source_depth"

declare -A configs
configs["cpg_only"]="search.allowed_sources='[cpg]' search.topk.retrieve=100 search.topk.rerank=25 search.search_mode=both"
configs["textbooks_only"]="search.allowed_sources='[textbooks]' search.topk.retrieve=100 search.topk.rerank=25 search.search_mode=both"
configs["fewer_docs"]="search.allowed_sources=all search.topk.retrieve=10 search.topk.rerank=5 search.search_mode=both"
configs["more_docs"]="search.allowed_sources=all search.topk.retrieve=200 search.topk.rerank=50 search.search_mode=both"

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