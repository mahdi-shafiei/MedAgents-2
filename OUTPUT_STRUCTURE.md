# MedAgents-2 Output Structure

This document describes the hierarchical output structure used by MedAgents-2 for organizing experiment results and ablation studies.

## 📁 Directory Structure

```
output/
├── medqa/                                # Task/Dataset name
│   ├── baseline/                         # Experiment name (ablation name)
│   │   ├── run_0/                        # Run number
│   │   │   ├── gpt4o_mini/               # Base model
│   │   │   │   ├── config.yaml           # Full configuration snapshot
│   │   │   │   ├── results.json          # Experiment results
│   │   │   │   └── logs/                 # Log files
│   │   │   ├── gpt4o/
│   │   │   │   ├── config.yaml
│   │   │   │   ├── results.json
│   │   │   │   └── logs/
│   │   │   └── claude3_sonnet/
│   │   ├── run_1/
│   │   │   ├── gpt4o_mini/
│   │   │   └── gpt4o/
│   │   └── run_2/
│   │       └── gpt4o_mini/
│   ├── no_rewrite/                       # Experiment name
│   │   ├── run_0/
│   │   │   └── gpt4o_mini/
│   │   ├── run_1/
│   │   └── run_2/
│   ├── no_review/                        # Experiment name
│   │   ├── run_0/
│   │   │   └── gpt4o_mini/
│   │   └── run_1/
│   └── easy_2_experts_1_round/           # Experiment name
│       ├── run_0/
│       └── run_1/
├── medmcqa/                              # Task/Dataset name
│   ├── baseline/
│   │   ├── run_0/
│   │   │   └── gpt4o_mini/
│   │   └── run_1/
│   └── no_rewrite/
│       └── run_0/
└── pubmedqa/
    └── baseline/
        └── run_0/
            └── gpt4o_mini/
```

## 🏗️ Hierarchy Levels

### 1. **Task Level** (`medqa/`, `medmcqa/`, `pubmedqa/`)
- Represents the dataset or task being evaluated
- Each dataset gets its own top-level directory

### 2. **Experiment Level** (`baseline/`, `no_rewrite/`, `no_review/`)
- Represents different ablation studies or configurations
- Names are specified via `execution.experiment_name` configuration
- Examples:
  - `baseline`: Default configuration
  - `no_rewrite`: Query rewriting disabled
  - `no_review`: Document review disabled
  - `easy_2_experts_1_round`: Triage disabled, easy level
  - `1_expert_1_round`: Custom configuration

### 3. **Run Level** (`run_0/`, `run_1/`, `run_2/`)
- Represents multiple runs of the same experiment
- Used for statistical significance and reproducibility
- Run IDs are configurable via `execution.experiments.run_id`

### 4. **Model Level** (`gpt4o_mini/`, `gpt4o/`, `claude3_sonnet/`)
- Represents different base models used
- Model names are derived from `model.name` configuration
- Hyphens are replaced with underscores for directory compatibility

## 📄 File Contents

### `config.yaml`
- Complete configuration snapshot for the specific run
- Includes all parameters: model, search, triage, execution settings
- Ensures full reproducibility

### `results.json`
- Structured results from the experiment
- Contains:
  - Question and options
  - Final answer and confidence scores
  - Expert details and reasoning
  - Token usage statistics
  - Timing information
  - Full conversation logs

### `logs/`
- Directory for log files
- Contains detailed execution logs
- Useful for debugging and analysis

## 🔬 Experiment Naming Convention

Experiment names are specified via the `execution.experiment_name` configuration parameter. This allows for clear, descriptive names that indicate what the experiment tests.

### Common Experiment Names

#### Search Ablations
- `baseline`: All search features enabled
- `no_rewrite`: Query rewriting disabled
- `no_review`: Document review disabled
- `no_rewrite_no_review`: Both features disabled
- `web_only`: Web search only
- `vector_only`: Vector search only
- `search_history_shared`: Shared search history
- `search_history_none`: No search history

#### Triage Ablations
- `baseline`: Triage enabled (default)
- `easy_2_experts_1_round`: Triage disabled, forced easy level
- `medium_3_experts_2_rounds`: Triage disabled, forced medium level
- `hard_3_experts_3_rounds`: Triage disabled, forced hard level
- `1_expert_1_round`: Custom configuration with 1 expert, 1 round
- `3_experts_2_rounds`: Custom configuration with 3 experts, 2 rounds

## 🚀 Usage Examples

### Running Baseline Experiment
```bash
python run_experiments.py \
    dataset.name=medqa \
    execution.experiments.run_id=0 \
    execution.experiment_name=baseline
# Creates: output/medqa/baseline/run_0/gpt4o_mini/
```

### Running Ablation Study
```bash
python run_experiments.py \
    dataset.name=medqa \
    execution.experiments.run_id=0 \
    execution.experiment_name=no_rewrite \
    search.rewrite=false
# Creates: output/medqa/no_rewrite/run_0/gpt4o_mini/
```

### Running with Different Model
```bash
python run_experiments.py \
    dataset.name=medqa \
    execution.experiments.run_id=0 \
    execution.experiment_name=baseline \
    model.name=gpt-4o
# Creates: output/medqa/baseline/run_0/gpt4o/
```

### Running Multiple Runs
```bash
python run_experiments.py dataset.name=medqa execution.experiments.run_id=0 execution.experiment_name=baseline
python run_experiments.py dataset.name=medqa execution.experiments.run_id=1 execution.experiment_name=baseline
python run_experiments.py dataset.name=medqa execution.experiments.run_id=2 execution.experiment_name=baseline
# Creates: output/medqa/baseline/run_0/, run_1/, run_2/
```

### Running Triage Ablation
```bash
python run_experiments.py \
    dataset.name=medqa \
    execution.experiments.run_id=0 \
    execution.experiment_name=easy_2_experts_1_round \
    triage.disable_triage=true \
    triage.forced_level=easy
# Creates: output/medqa/easy_2_experts_1_round/run_0/gpt4o_mini/
```

### Running Custom Configuration
```bash
python run_experiments.py \
    dataset.name=medqa \
    execution.experiments.run_id=0 \
    execution.experiment_name=1_expert_1_round \
    triage.disable_triage=true \
    triage.forced_level=custom \
    triage.forced_level_custom.num_experts=1 \
    triage.forced_level_custom.max_rounds=1 \
    triage.forced_level_custom.search_mode=none
# Creates: output/medqa/1_expert_1_round/run_0/gpt4o_mini/
```

## 🎯 Benefits

1. **Self-Documenting**: Directory structure clearly shows experiment hierarchy
2. **Scalable**: Can handle hundreds of experiments across multiple datasets
3. **Reproducible**: Full config saved with each run
4. **Flexible**: Easy to add new tasks, experiments, runs, or models
5. **Clear**: Intuitive navigation and understanding of experiment relationships
6. **Efficient**: No unnecessary registry files, structure serves as metadata
7. **Simple**: Uses command-line overrides instead of multiple config files

## 📊 Analysis

The structure makes it easy to:
- Compare results across different experiments
- Aggregate results across multiple runs
- Analyze performance across different models
- Track ablation study progress
- Generate summary statistics and visualizations 