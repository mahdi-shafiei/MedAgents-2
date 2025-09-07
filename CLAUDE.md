# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running Experiments
```bash
# Interactive mode
python main.py

# Single question mode
python main.py --mode single --question "A patient presents with..." --options "A:Option1" "B:Option2"

# Batch processing
python main.py --mode batch --questions questions.json --output results.json --difficulty medium

# Run experiments with Hydra configuration
python run_experiments.py execution.dataset.name=medqa execution.dataset.split=test_hard execution.model.name=gpt-4o-mini
```

### Automated Experiment Scripts
```bash
# Basic EBMedAgents experiments across all datasets
./run_ebagents.sh

# Run triage ablation studies
./run_triage_ablation.sh

# Run search/retrieval ablation studies  
./run_search_ablation.sh

# Run orchestration ablation studies
./run_orchestrate_ablation.sh

# Run MedRAG-style experiments
./run_medrag.sh

# Run hard triage experiments
./run_triage_hard.sh
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp .example.env .env
# Edit .env with your OPENAI_API_KEY, OPENAI_ENDPOINT, and MILVUS_URI

# Code formatting (if making contributions)
pip install black
black .
```

## Architecture Overview

MedAgents-2 is a multi-agent medical question answering system with four main components:

### Core Agents
- **Triage Agent** (`triage.py`): Analyzes question complexity, selects appropriate experts, creates detailed expert profiles with medical specializations
- **Expert Agents** (`expert.py`): Medical specialists that research questions using search tools and provide structured answers with reasoning and confidence scores
- **Search Agent** (`search_tools.py`, `retriever.py`): Configurable search engine supporting web search and vector search (Milvus) across medical literature and textbooks
- **Orchestrator** (`orchestrate.py`): Manages multi-agent discussions, synthesizes expert opinions, provides feedback, and determines consensus

### System Entry Points
- **Main Interface** (`main.py`): Interactive and batch processing interface for single questions or datasets
- **Experiment Runner** (`run_experiments.py`): Hydra-based experiment execution with extensive configuration options
- **EBMedAgents** (`ebagents.py`): Core orchestration class that coordinates all agents

### Configuration System
Uses Hydra for flexible configuration with configs in `conf/`:
- `config.yaml`: Main configuration entry point
- `triage/`: Triage agent settings, expert selection, difficulty assessment
- `search/`: Search configuration (web/vector modes, query rewriting, document review)
- `orchestrate/`: Discussion modes, multi-round settings, agent coordination
- `execution/`: Dataset selection, model settings, experiment parameters

### Key Features
- **Multi-modal Search**: Combines web search and vector database (Milvus) retrieval
- **Dynamic Expert Selection**: Triage creates specialized expert profiles based on question complexity
- **Evidence-based Reasoning**: All answers include citations, confidence scores, and structured reasoning
- **Ablation Study Support**: Extensive configuration options for controlled experiments
- **Multiple Discussion Modes**: Group chat, individual responses, voting mechanisms

### Experimental Framework
The system supports comprehensive ablation studies:
- Triage vs no-triage comparisons
- Search modality ablations (web vs vector vs both)
- Agent configuration studies (1-5 agents, 1-3 rounds)
- Search feature ablations (query rewrite, document review)
- MedRAG baseline comparisons

### Data Flow
1. Question enters via `main.py` or `run_experiments.py`
2. Triage agent assesses complexity and selects experts
3. Expert agents research using search tools and provide initial answers
4. Orchestrator facilitates multi-round discussions and feedback
5. System generates final answer with evidence and reasoning
6. Results saved in hierarchical output structure: `output/{dataset}/{experiment}/{run_id}/{model}/`

## Important Notes
- Environment variables must be configured in `.env` for OpenAI API and Milvus connection
- The system uses async/await patterns extensively with `nest_asyncio` for Jupyter compatibility
- All agents use OpenAI's Agent SDK with structured Pydantic models for responses
- Search tools support configurable sources (CPG guidelines, textbooks, etc.)
- Results are automatically saved with comprehensive metadata for reproducibility