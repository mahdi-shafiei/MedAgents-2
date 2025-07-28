# Orchestrate Configuration & Discussion Mode Ablation Experiments

This directory contains the default and override configs for the orchestrate (agent coordination) component of MedAgents-2.

## 🔬 Discussion Mode Ablation Study

**Goal:** Assess the fundamental impact of different agent interaction patterns on decision quality and efficiency.

| Experiment Name              | discussion_mode              | max_rounds | log_intermediate_steps | Notes                                    |
|-----------------------------|------------------------------|------------|----------------------|------------------------------------------|
| Group Chat with Orchestrator| group_chat_with_orchestrator | 3          | true                 | Full interaction: experts see each other + orchestrator feedback |
| Group Chat Voting Only      | group_chat_voting_only       | 3          | true                 | Experts see each other, no orchestrator guidance |
| One-on-One Sync             | one_on_one_sync              | 3          | true                 | Experts only get orchestrator feedback, no peer interaction |
| Independent (Baseline)      | independent                  | 3          | true                 | No interaction between experts or orchestrator |

---

## 🏃‍♂️ How to Run the Ablation Study

Override the discussion mode parameter at runtime using Hydra/OmegaConf:

```bash
# Run each experiment
python main.py orchestrate.discussion_mode=group_chat_with_orchestrator
python main.py orchestrate.discussion_mode=group_chat_voting_only
python main.py orchestrate.discussion_mode=one_on_one_sync
python main.py orchestrate.discussion_mode=independent
```

Or use minimal override configs for frequently repeated experiments.

---

## 📊 Expected Outcomes

- **`group_chat_with_orchestrator`**: Highest consensus quality, moderate token usage
- **`group_chat_voting_only`**: Good consensus, lower token usage (no orchestrator)
- **`one_on_one_sync`**: Moderate consensus, focused expert development  
- **`independent`**: Baseline performance, minimal token usage

---

**Tip:**
- Use `log_intermediate_steps=true` for detailed analysis of discussion progression
- Monitor token usage patterns across different modes
- For production use, `group_chat_with_orchestrator` provides the best balance of quality and coordination 