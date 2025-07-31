# Search Configuration & Ablation Experiments

This directory contains the default and override configs for the search (retrieval) component of MedAgents-2.

## 🔬 Search Ablation Studies

Based on the experimental scripts, we conduct several ablation studies to understand the impact of search mechanisms:

---

## **Group 1: Search Features Ablation**
**Goal:** Assess the contribution of individual search pipeline features (query rewrite, document review).

| Experiment Name         | rewrite | review | search_mode | topk.retrieve | topk.rerank | Notes                        |
|------------------------|---------|--------|-------------|---------------|-------------|------------------------------|
| Baseline               | true    | true   | both        | 10            | 5           | All features enabled         |
| No Query Rewrite       | false   | true   | both        | 10            | 5           | Disable query rewriting      |
| No Document Review     | true    | false  | both        | 10            | 5           | Disable document review      |
| No Rewrite & No Review | false   | false  | both        | 10            | 5           | Both features disabled       |

**Script:** `run_search_ablation.sh` (Experiment 1)

---

## **Group 2: Search Modality Ablation**
**Goal:** Isolate the effect of different search modalities (web, vector, both).

| Experiment Name   | search_mode | rewrite | review | Notes                    |
|------------------|-------------|---------|--------|--------------------------|
| Web Only         | web         | true    | true   | Only web search enabled  |
| Vector Only      | vector      | true    | true   | Only vector search       |
| Both             | both        | true    | true   | Both modalities enabled  |

**Script:** `run_search_ablation.sh` (Experiment 2)

---

## **Group 3: Search History Ablation**
**Goal:** Evaluate the impact of search history sharing strategies among agents.

| Experiment Name   | search_history | search_mode | rewrite | review | Notes                        |
|------------------|----------------|-------------|---------|--------|------------------------------|
| Individual       | individual     | both        | true    | true   | Each agent has own history   |
| Shared           | shared         | both        | true    | true   | Agents share search history  |

**Script:** `run_search_ablation.sh` (Experiment 3)

---

## **Group 4: Search Source Depth Ablation**
**Goal:** Assess the effect of source restriction and retrieval depth.

| Experiment Name   | allowed_sources | topk.retrieve | topk.rerank | search_mode | rewrite | review | Notes                        |
|------------------|-----------------|---------------|-------------|-------------|---------|--------|------------------------------|
| CPG Only         | [cpg]           | 100           | 25          | both        | true    | true   | Only CPG as source           |
| Textbooks Only   | [textbooks]     | 100           | 25          | both        | true    | true   | Only textbooks as source     |
| Fewer Docs       | all             | 10            | 5           | both        | true    | true   | Retrieve fewer documents     |
| More Docs        | all             | 200           | 50          | both        | true    | true   | Retrieve more documents      |

**Script:** `run_search_ablation.sh` (Experiment 4)

---

## **Group 5: MedRAG Search Configuration**
**Goal:** Test simplified search configurations for MedRAG-style experiments.

| Experiment Name | search_mode | rewrite | review | Notes                        |
|----------------|-------------|---------|--------|------------------------------|
| MedRAG Search   | vector       | false   | false  | Simplified vector-only search |
| iMedRAG Search  | vector       | false   | false  | Same as MedRAG but 3 rounds  |

**Script:** `run_medrag.sh`

---

## 🏃‍♂️ How to Run Each Experiment

### Search Features Ablation
```bash
# Baseline
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_features/baseline search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=true search.review=true search.search_mode=both

# No Query Rewrite
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_features/no_query_rewrite search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=false search.review=true search.search_mode=both

# No Document Review
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_features/no_document_review search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=true search.review=false search.search_mode=both

# No Rewrite & No Review
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_features/no_rewrite_no_review search.topk.retrieve=10 search.topk.rerank=5 search.rewrite=false search.review=false search.search_mode=both
```

### Search Modality Ablation
```bash
# Web Only
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_modality/web_only search.search_mode=web

# Vector Only
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_modality/vector_only search.search_mode=vector

# Both
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_modality/both search.search_mode=both
```

### Search History Ablation
```bash
# Individual History
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_history/individual search.search_history=individual search.search_mode=both

# Shared History
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_history/shared search.search_history=shared search.search_mode=both
```

### Search Source Depth Ablation
```bash
# CPG Only
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_source_depth/cpg_only search.allowed_sources='[cpg]' search.topk.retrieve=100 search.topk.rerank=25 search.search_mode=both

# Textbooks Only
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_source_depth/textbooks_only search.allowed_sources='[textbooks]' search.topk.retrieve=100 search.topk.rerank=25 search.search_mode=both

# Fewer Docs
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_source_depth/fewer_docs search.allowed_sources=all search.topk.retrieve=10 search.topk.rerank=5 search.search_mode=both

# More Docs
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini execution.experiments.run_id=0 execution.experiment_name=search_source_depth/more_docs search.allowed_sources=all search.topk.retrieve=200 search.topk.rerank=50 search.search_mode=both
```

---

## 📊 Experimental Setup

**Dataset:** medqa (primary for search ablations)
**Split:** test_hard
**Model:** gpt-4o-mini
**Run IDs:** 0 (for search ablations), 0, 1, 2 (for MedRAG)

---

**Tip:**
- Use the provided shell scripts for automated experiment execution
- Monitor search quality and retrieval performance across different configurations
- Focus on the most impactful search features based on your research goals 