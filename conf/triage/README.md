# Triage Configuration & Ablation Experiments

This directory contains the default and override configs for the triage (difficulty/agent role) component of MedAgents-2.

## 🔬 Streamlined Triage Ablation Experiments

We recommend focusing on the most impactful ablations:

---

## **Group 1: Triage vs. No Triage**
**Goal:** Assess the impact of using the triage agent versus bypassing it.

| Experiment Name   | disable_triage | forced_level         | Notes                        |
|------------------|----------------|----------------------|------------------------------|
| Triage Enabled   | false          | -                    | Use full triage agent        |
| Triage Disabled  | true           | easy/medium/hard     | Force a fixed level |

You can use `forced_level: easy`, `forced_level: medium`, `forced_level: hard` to help with your ablation study.

---

## **Group 2: When Triage is Disabled — Number of Agents, Rounds, and Tool Use**
**Goal:** When triage is disabled, ablate on the number of agents, number of rounds, and whether tool use (search) is enabled. For each forced level, you can set these parameters independently.

| forced_level | num_experts | max_rounds | search_mode | Example Command                                                                 |
|--------------|-------------|------------|-------------|--------------------------------------------------------------------------------|
| easy         | 1           | 1          | none        | python main.py triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=1 triage.easy.search_mode=none |
| easy         | 3           | 2          | required    | python main.py triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=3 triage.easy.max_rounds=2 triage.easy.search_mode=required |
| custom       | 2           | 3          | required    | python main.py triage.disable_triage=true triage.forced_level=custom triage.forced_level_custom.num_experts=2 triage.forced_level_custom.max_rounds=3 triage.forced_level_custom.search_mode=required |

You can set any values in `forced_level_custom` for fully custom ablation.

---

## 🏃‍♂️ How to Run Each Experiment

Override the relevant parameters at runtime using Hydra/OmegaConf, e.g.:

```bash
# Triage enabled (default)
python main.py

# Triage disabled, 1 agent, 1 round, no tool
python main.py triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=1 triage.easy.search_mode=none

# Triage disabled, custom config: 2 agents, 3 rounds, tool enabled
python main.py triage.disable_triage=true triage.forced_level=custom triage.forced_level_custom.num_experts=2 triage.forced_level_custom.max_rounds=3 triage.forced_level_custom.search_mode=required
```

Or use minimal override configs for frequently repeated ablations.

---

**Tip:**
- Only create override config files for ablations you want to frequently reuse.
- For most experiments, use command-line overrides for flexibility. 