# Search Configuration & Ablation Experiments

This directory contains the default and override configs for the search (retrieval) component of MedAgents-2.

## 🔬 Grouped Search Ablation Experiments (Paper-Style)

Below, we organize ablation experiments into thematic groups, each with a clear motivation and a table of settings. This structure is suitable for a scientific paper's methods section.

---

## **Group 1: Search Feature Ablations**
**Goal:** Assess the contribution of individual search pipeline features (query rewrite, document review).

| Experiment Name         | rewrite | review | search_mode | search_history | Notes                        |
|------------------------|---------|--------|-------------|---------------|------------------------------|
| Baseline               | true    | true   | both        | individual    | All features enabled         |
| No Query Rewrite       | false   | true   | both        | individual    | Disable query rewriting      |
| No Document Review     | true    | false  | both        | individual    | Disable document review      |
| No Rewrite & No Review | false   | false  | both        | individual    | Both features disabled       |

---

## **Group 2: Search Modality Ablations**
**Goal:** Isolate the effect of different search modalities (web, vector, both).

| Experiment Name   | search_mode | rewrite | review | search_history | Notes                    |
|------------------|-------------|---------|--------|---------------|--------------------------|
| Web Only         | web         | true    | true   | individual    | Only web search enabled  |
| Vector Only      | vector      | true    | true   | individual    | Only vector search       |
| Both             | both        | true    | true   | individual    | Both modalities enabled  |

---

## **Group 3: Search History Sharing**
**Goal:** Evaluate the impact of search history sharing strategies among agents.

| Experiment Name   | search_history | search_mode | rewrite | review | Notes                        |
|------------------|----------------|-------------|---------|--------|------------------------------|
| Individual       | individual     | both        | true    | true   | Each agent has own history   |
| Shared           | shared         | both        | true    | true   | Agents share search history  |
| None             | none           | both        | true    | true   | No search history used       |

---

## **Group 4: Source and Retrieval Depth**
**Goal:** Assess the effect of source restriction and retrieval depth.

| Experiment Name   | allowed_sources | topk.retrieve | topk.rerank | search_mode | rewrite | review | Notes                        |
|------------------|-----------------|---------------|-------------|-------------|---------|--------|------------------------------|
| CPG Only         | [cpg]           | 100           | 25          | both        | true    | true   | Only CPG as source           |
| Textbooks Only   | [textbooks]     | 100           | 25          | both        | true    | true   | Only textbooks as source     |
| Fewer Docs       | all             | 10            | 5           | both        | true    | true   | Retrieve fewer documents     |
| More Docs        | all             | 200           | 50          | both        | true    | true   | Retrieve more documents      |

---

## 🏃‍♂️ How to Run Each Experiment

Override the relevant parameters at runtime using Hydra/OmegaConf, e.g.:

```bash
python main.py search.rewrite=false search.review=false
python main.py search.search_mode=web
python main.py search.search_history=shared
python main.py search.allowed_sources='[cpg]'
python main.py search.topk.retrieve=10 search.topk.rerank=5
```

Or use minimal override configs for frequently repeated ablations.

---

**Tip:**
- Only create override config files for ablations you want to frequently reuse.
- For most experiments, use command-line overrides for flexibility. 