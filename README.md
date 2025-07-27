# MedAgents-2: Multi-Agent Medical Question Answering System

MedAgents-2 is a next-generation, evidence-based multi-agent system for medical question answering. It brings together a team of specialized AI agents—each with unique expertise—to collaboratively solve complex medical problems through structured debate, dynamic evidence gathering, and consensus-building.

---

## 🏗️ How MedAgents-2 Works

Imagine a virtual medical boardroom, where each agent plays a distinct, expert role:

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  The Triage   │    │  The Experts  │    │ The Moderator │
│  Coordinator  │    │  Panel        │    │               │
│               │    │               │    │               │
│ • Reads and   │───▶│ • Specialists │───▶│ • Guides      │
│   classifies  │    │   debate      │    │   discussion  │
│   questions   │    │ • Gather and  │    │   interpret   │
│ • Selects     │    │   evidence    │    │   evidence    │
│   the right   │    │ • Justify     │    │   answers     │
│   experts     │    │               │    │               │
│ • Assesses    │    │               │    │               │
│   complexity  │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                  ┌────────────────────┐
                  │  The Researcher    │
                  │  (Search Engine)   │
                  │                    │
                  │ • Finds evidence   │
                  │ • Evaluates        │
                  │   relevance        │
                  │ • Refines queries  │
                  └────────────────────┘
```

---

## 🔑 Meet the Team

**The Triage Coordinator**  
Reads each question, determines its complexity, and selects the most relevant medical specialists for the case. Crafts detailed, CV-style profiles for each expert, ensuring the right knowledge is at the table. Can also run in “ablation” mode for controlled experiments.

**The Expert Panel**  
A diverse group of AI medical specialists, each with a unique background and research focus. They analyze the question, use advanced search tools to gather evidence, and provide structured, explainable answers—complete with reasoning, confidence, and justification. They can review previous searches and adapt their approach as the discussion evolves.

**The Researcher**  
A powerful, configurable search engine that scours medical literature, clinical guidelines, and textbooks. It rewrites queries for better results, checks for redundant searches, and adapts its strategy based on the needs of the experts. Integrates with Milvus for vector search and supports multiple medical sources.

**The Moderator**  
Guides the debate, synthesizes expert opinions, and provides actionable feedback. Highlights areas of agreement and disagreement, suggests further exploration, and decides when consensus is reached or more discussion is needed. Ensures every decision is well-justified and evidence-based.

---

## 🚀 Quick Start

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**  
   Copy `.example.env` to `.env` and fill in your credentials.

3. **Run in interactive mode**  
   ```bash
   python main.py
   ```

4. **Single question mode**  
   ```bash
   python main.py --mode single \
     --question "A patient presents with chest pain..." \
     --options "A:Angina" "B:Heart attack" "C:Indigestion" "D:Anxiety"
   ```

5. **Batch processing**  
   ```bash
   python main.py --mode batch \
     --questions questions.json \
     --output results.json \
     --difficulty medium
   ```

---

## ⚙️ Configuration

MedAgents-2 uses Hydra for flexible configuration. Main config files:
- `conf/config.yaml`: Main entry point
- `conf/orchestrate/default.yaml`: Agent orchestration
- `conf/triage/default.yaml`: Triage/difficulty
- `conf/search/default.yaml`: Search/retrieval
- `conf/execution/default.yaml`: Execution

**Ablation experiments and advanced configuration:**  
- [Triage Config & Ablation](conf/triage/README.md)  
- [Search Config & Ablation](conf/search/README.md)

---

## 📝 More Info

- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
- See [LICENSE](LICENSE) for license details.

---

For detailed architecture, advanced features, and ablation studies, refer to the documentation in the `conf/` subfolders.