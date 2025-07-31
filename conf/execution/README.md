# Execution Configuration & Baseline Experiments

This directory contains the default and override configs for the execution (dataset, model, experiment management) component of MedAgents-2.

## 🔬 Baseline Experiment Configuration

Based on the experimental scripts, we conduct baseline experiments across multiple medical datasets:

---

## **Baseline Experiment Setup**
**Goal:** Establish baseline performance across multiple medical datasets using the default MedAgents-2 configuration.

| Experiment Name | Dataset | Split | Model | Run IDs | Notes                        |
|----------------|---------|-------|-------|---------|------------------------------|
| Baseline       | All 9 datasets | test_hard | gpt-4o-mini | 0, 1, 2 | Default MedAgents-2 configuration |

**Script:** `run_ebagents.sh`

---

## **Available Datasets**

The system supports the following medical datasets:

| Dataset Name | Description | Split Used |
|--------------|-------------|------------|
| medqa | Medical Question Answering | test_hard |
| medbullets | Medical Bullets | test_hard |
| medexqa | Medical Expert QA | test_hard |
| medmcqa | Medical Multiple Choice QA | test_hard |
| medxpertqa-r | Medical Expert QA (Random) | test_hard |
| medxpertqa-u | Medical Expert QA (Uniform) | test_hard |
| mmlu | Massive Multitask Language Understanding | test_hard |
| mmlu-pro | MMLU Professional | test_hard |
| pubmedqa | PubMed QA | test_hard |

---

## **Model Configuration**

| Model Name | Provider | Notes |
|------------|----------|-------|
| gpt-4o-mini | OpenAI | Primary model for experiments |

---

## 🏃‍♂️ How to Run Baseline Experiments

### Single Dataset Baseline
```bash
# Run baseline for a single dataset
python run_experiments.py \
    execution.dataset.name=medqa \
    execution.dataset.split=test_hard \
    execution.model.name=gpt-4o-mini \
    execution.experiments.run_id=0 \
    execution.experiment_name=ebagents/baseline
```

### Multi-Dataset Baseline
```bash
# Run baseline across all datasets (using script)
./run_ebagents.sh
```

### Custom Dataset Configuration
```bash
# Run with custom dataset
python run_experiments.py \
    execution.dataset.name=medqa \
    execution.dataset.split=test_hard \
    execution.model.name=gpt-4o-mini \
    execution.experiments.run_id=0 \
    execution.experiment_name=custom_experiment_name
```

---

## 📊 Experimental Setup

**Datasets:** medqa, medbullets, medexqa, medmcqa, medxpertqa-r, medxpertqa-u, mmlu, mmlu-pro, pubmedqa
**Split:** test_hard
**Model:** gpt-4o-mini
**Run IDs:** 0, 1, 2 (for statistical significance)
**Experiment Name:** ebagents/baseline

---

## 📁 Output Structure

Results are saved in the following structure:
```
output/
├── [DATASET]/
│   └── [EXPERIMENT_NAME]/
│       └── run_[RUN_ID]/
│           └── [MODEL]/
│               ├── results.json
│               ├── logs/
│               └── artifacts/
```

---

**Tip:**
- Use the provided shell scripts for automated experiment execution
- Monitor performance across different datasets to identify domain-specific patterns
- The baseline configuration serves as the foundation for all ablation studies 