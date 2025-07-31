# Orchestrate Configuration & Discussion Mode Ablation Experiments

This directory contains the default and override configs for the orchestrate (agent coordination) component of MedAgents-2.

## 🔬 Discussion Mode Ablation Study

**Goal:** Assess the fundamental impact of different agent interaction patterns on decision quality and efficiency.

| Experiment Name              | discussion_mode              | triage_config | search_mode | Notes                                    |
|-----------------------------|------------------------------|---------------|-------------|------------------------------------------|
| Independent                 | independent                  | disabled      | web         | No interaction between experts or orchestrator |
| Group Chat with Orchestrator| group_chat_with_orchestrator | enabled       | web         | Full interaction: experts see each other + orchestrator feedback |
| Group Chat Voting Only      | group_chat_voting_only       | enabled       | web         | Experts see each other, no orchestrator guidance |
| One-on-One Sync             | one_on_one_sync              | enabled       | web         | Experts only get orchestrator feedback, no peer interaction |

**Script:** `run_orchestrate_ablation.sh`

---

## **Additional Discussion Mode Configurations**

### MedRAG Discussion Mode
**Goal:** Test simplified discussion modes for MedRAG-style experiments.

| Experiment Name | discussion_mode              | triage_config | search_mode | Notes                        |
|----------------|------------------------------|---------------|-------------|------------------------------|
| MedRAG         | group_chat_voting_only       | disabled      | vector      | Single agent, voting only    |
| iMedRAG        | group_chat_voting_only       | disabled      | vector      | Same as MedRAG but 3 rounds  |

**Script:** `run_medrag.sh`

---

## 🏃‍♂️ How to Run Each Experiment

### Core Discussion Mode Ablation
```bash
# Independent (Baseline)
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=discussion_mode_ablation/independent orchestrate.discussion_mode=independent triage.disable_triage=true triage.forced_level=custom triage.forced_level_custom.num_experts=1 triage.forced_level_custom.max_rounds=3 search.search_mode=web

# Group Chat with Orchestrator
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=discussion_mode_ablation/group_chat_with_orchestrator orchestrate.discussion_mode=group_chat_with_orchestrator search.search_mode=web

# Group Chat Voting Only
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=discussion_mode_ablation/group_chat_voting_only orchestrate.discussion_mode=group_chat_voting_only search.search_mode=web

# One-on-One Sync
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=discussion_mode_ablation/one_on_one_sync orchestrate.discussion_mode=one_on_one_sync search.search_mode=web
```

### MedRAG Discussion Mode
```bash
# MedRAG: group_chat_voting_only with vector search
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=medrag/medrag triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=1 triage.easy.search_mode=required search.rewrite=false search.review=false search.search_mode=vector orchestrate.discussion_mode=group_chat_voting_only

# iMedRAG: same discussion mode but 3 rounds
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=medrag/imedrag triage.disable_triage=true triage.forced_level=easy triage.easy.num_experts=1 triage.easy.max_rounds=3 triage.easy.search_mode=required search.rewrite=false search.review=false search.search_mode=vector orchestrate.discussion_mode=group_chat_voting_only
```

---

## 📊 Experimental Setup

**Dataset:** medqa (primary for orchestrate ablations)
**Split:** test_hard
**Model:** gpt-4o-mini
**Run IDs:** 0 (for orchestrate ablations), 0, 1, 2 (for MedRAG)

---

## 📈 Expected Outcomes

- **`independent`**: Baseline performance, minimal token usage, no coordination overhead
- **`group_chat_with_orchestrator`**: Highest consensus quality, moderate token usage, full coordination
- **`group_chat_voting_only`**: Good consensus, lower token usage (no orchestrator guidance)
- **`one_on_one_sync`**: Moderate consensus, focused expert development, orchestrator-only guidance

---

**Tip:**
- Use the provided shell scripts for automated experiment execution
- Monitor token usage patterns across different discussion modes
- Focus on the discussion mode that provides the best balance of quality and efficiency for your use case 