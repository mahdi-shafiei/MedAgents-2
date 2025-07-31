# Triage Configuration & Ablation Experiments

This directory contains the default and override configs for the triage (difficulty/agent role) component of MedAgents-2.

## 🔬 Triage Ablation Studies

Based on the experimental scripts, we conduct several ablation studies to understand the impact of triage mechanisms:

---

## **Group 1: Triage vs. No Triage (Basic Comparison)**
**Goal:** Assess the fundamental impact of using the triage agent versus bypassing it.

| Experiment Name   | disable_triage | forced_level | Notes                        |
|------------------|----------------|--------------|------------------------------|
| Enable Triage    | false          | hard         | Use full triage agent        |
| Disable Triage   | true           | hard         | Force hard level without triage |

**Script:** `run_triage_ablation.sh` (Group 3)

---

## **Group 2: Agent Configuration with Triage Disabled**
**Goal:** When triage is disabled, ablate on the number of agents, number of rounds, and search mode.

| Experiment Name              | num_experts | max_rounds | search_mode | Notes                        |
|-----------------------------|-------------|------------|-------------|------------------------------|
| 1 Agent, 1 Round, No Search | 1           | 1          | none        | Minimal configuration        |
| 1 Agent, 1 Round, With Search | 1        | 1          | required    | Single agent with search     |
| 1 Agent, 3 Rounds, With Search | 1        | 3          | required    | Single agent, multiple rounds |
| 2 Agents, 3 Rounds, With Search | 2        | 3          | required    | Multiple agents, multiple rounds |
| 3 Agents, 1 Round, With Search | 3        | 1          | required    | Multiple agents, single round |
| 3 Agents, 2 Rounds, With Search | 3        | 2          | required    | Multiple agents, moderate rounds |
| 3 Agents, 3 Rounds, With Search | 3        | 3          | required    | Full multi-agent setup       |
| 5 Agents, 1 Round, With Search | 5        | 1          | required    | Many agents, single round    |

**Script:** `run_triage_ablation.sh` (Group 2 - commented out)

---

## **Group 3: MedRAG Configuration**
**Goal:** Test simplified MedRAG-style configurations with minimal agent setup.

| Experiment Name | num_experts | max_rounds | search_mode | discussion_mode | Notes                        |
|----------------|-------------|------------|-------------|-----------------|------------------------------|
| MedRAG         | 1           | 1          | required    | group_chat_voting_only | Single agent, single round |
| iMedRAG        | 1           | 3          | required    | group_chat_voting_only | Single agent, multiple rounds |

**Script:** `run_medrag.sh`

---

## **Group 4: Multi-Dataset Hard Triage**
**Goal:** Test triage configuration across multiple medical datasets.

| Experiment Name | forced_level | search_mode | Datasets | Notes                        |
|----------------|--------------|-------------|----------|------------------------------|
| New            | hard         | web         | All 9 datasets | Hard triage across all datasets |

**Script:** `run_triage_hard.sh`

---

## 🏃‍♂️ How to Run Each Experiment

### Basic Triage Comparison
```bash
# Enable triage (default)
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=triage_configuration/enable_triage triage.forced_level=hard search.search_mode=web

# Disable triage
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=triage_configuration/disable_triage triage.disable_triage=true triage.forced_level=hard search.search_mode=web
```

### MedRAG Configuration
```bash
# MedRAG: 1 agent, 1 round
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=medrag/medrag triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=1 triage.easy.search_mode=required search.rewrite=false search.review=false search.search_mode=vector orchestrate.discussion_mode=group_chat_voting_only

# iMedRAG: 1 agent, 3 rounds
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=medrag/imedrag triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=3 triage.easy.search_mode=required search.rewrite=false search.review=false search.search_mode=vector orchestrate.discussion_mode=group_chat_voting_only
```

### Multi-Dataset Hard Triage
```bash
# Run across all datasets
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=ebagents/new triage.forced_level=hard search.search_mode=web
```

---

## 📊 Experimental Setup

**Datasets:** medqa, medbullets, medexqa, medmcqa, medxpertqa-r, medxpertqa-u, mmlu, mmlu-pro, pubmedqa
**Split:** test_hard
**Model:** gpt-4o-mini
**Run IDs:** 0, 1, 2 (for statistical significance)

---

**Tip:**
- Use the provided shell scripts for automated experiment execution
- Monitor performance across different triage configurations
- Focus on the most impactful ablations based on your research goals 