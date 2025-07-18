# MedAgents-2: Multi-Agent Medical Question Answering System

MedAgents-2 is an advanced multi-agent system for medical question answering that leverages specialized AI agents to collaboratively solve complex medical problems through structured debate and consensus-building.

## 🏗️ Architecture Overview

MedAgents-2 implements an **Evidence-Based Multi-Agent Architecture** with the following key components:

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Triage Unit   │    │  Expert Agents  │    │ Orchestrator    │
│                 │    │                 │    │                 │
│ • Classifies    │───▶│ • Domain        │───▶│ • Moderates     │
│   questions     │    │   specialists   │    │   discussions   │
│ • Selects       │    │ • Generate      │    │ • Builds        │
│   experts       │    │   responses     │    │   consensus     │
│ • Assesses      │    │ • Search        │    │ • Makes final   │
│   difficulty    │    │   evidence      │    │   decisions     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Search Unit    │
                    │                 │
                    │ • Retrieves     │
                    │   documents     │
                    │ • Evaluates     │
                    │   relevance     │
                    │ • Rewrites      │
                    │   queries       │
                    └─────────────────┘
```

## 🔧 Agent Components

### 1. Triage Unit (`triage.py`)
The **Triage Unit** is responsible for initial question analysis and expert selection:

- **Question Classification**: Analyzes medical questions to determine relevant specialties
- **Expert Selection**: Chooses appropriate medical experts based on question content
- **Difficulty Assessment**: Evaluates question complexity (easy/medium/hard)
- **Expert Profile Generation**: Creates detailed CV-style profiles for each selected expert

**Key Features:**
- Dynamic expert selection from 47+ medical specialties
- Weighted expert assignment based on relevance
- Automatic difficulty assessment
- Rich expert profiles with job titles, experience, and research focus

### 2. Expert Agents (`expert.py`)
**Expert Agents** are specialized medical professionals that:

- **Domain Expertise**: Represent specific medical specialties (Cardiology, Neurology, etc.)
- **Evidence Gathering**: Use search tools to find relevant medical information
- **Reasoned Responses**: Provide structured answers with confidence levels
- **Context Awareness**: Adapt responses based on discussion context

**Expert Response Format:**
```json
{
  "thought": "Reasoning process...",
  "answer": "A",
  "confidence": "high|medium|low",
  "justification": "Detailed explanation..."
}
```

### 3. Search Unit (`search_tools.py`)
The **Search Unit** provides evidence-based support:

- **Document Retrieval**: Searches medical knowledge bases
- **Query Rewriting**: Optimizes search queries for better results
- **Relevance Evaluation**: Filters and ranks retrieved documents
- **Adaptive RAG**: Determines when additional information is needed

**Search Strategies:**
- `independent`: Each expert has separate search history
- `cross_talk`: Experts share search context and can see others' queries
- `previous_answers`: Experts see their own previous responses

### 4. Orchestrator (`orchestrate.py`)
The **Orchestrator** manages the multi-agent discussion:

- **Discussion Moderation**: Facilitates expert interactions
- **Consensus Building**: Weighs expert opinions and builds agreement
- **Decision Making**: Makes final decisions based on expert input
- **Progress Tracking**: Monitors discussion progress and determines when to continue

**Orchestrator Features:**
- Weighted voting system with confidence adjustments
- Expert feedback generation
- Discussion continuation assessment
- Final decision synthesis

## 🚀 Usage

### Quick Start

1. **Interactive Mode** (Default):
```bash
python main.py
```

2. **Single Question Mode**:
```bash
python main.py --mode single \
  --question "A patient presents with chest pain..." \
  --options "A:Angina" "B:Heart attack" "C:Indigestion" "D:Anxiety"
```

3. **Batch Processing**:
```bash
python main.py --mode batch \
  --questions questions.json \
  --output results.json \
  --difficulty medium
```

### Configuration

MedAgents-2 uses Hydra for configuration management. Key configuration files:

- `conf/config.yaml`: Main configuration
- `conf/model/default.yaml`: LLM settings
- `conf/orchestration/default.yaml`: Agent orchestration settings
- `conf/difficulty/default.yaml`: Difficulty-based parameters

### Environment Variables

Create a `.env` file with:
```env
OPENAI_ENDPOINT=your_openai_endpoint
OPENAI_API_KEY=your_api_key
MILVUS_URI=your_milvus_uri  # For document retrieval
```

## 📊 Difficulty Levels

The system automatically adapts based on question difficulty:

| Difficulty | Experts | Rounds | Knowledge Gathering |
|------------|---------|--------|-------------------|
| Easy       | 2       | 1      | No                |
| Medium     | 3       | 2      | Yes               |
| Hard       | 3       | 3      | Yes               |

## 🔄 Discussion Flow

1. **Triage Phase**: Question analysis and expert selection
2. **Knowledge Gathering**: Pre-debate evidence collection (optional)
3. **Multi-Round Debate**: 
   - Experts provide responses
   - Orchestrator analyzes and provides feedback
   - Discussion continues until consensus or max rounds
4. **Final Decision**: Weighted consensus with justification

## 📈 Monitoring and Logging

The system provides comprehensive logging:

- **Token Usage**: Detailed tracking of input/output tokens
- **Expert Responses**: Complete expert analysis and reasoning
- **Discussion History**: Full debate transcript with orchestrator feedback
- **Performance Metrics**: Response times, consensus quality, etc.

### Example Output Structure:
```json
{
  "rounds": [
    {
      "round_num": 0,
      "expert_results": [...],
      "current_decision": {...},
      "orchestrator_feedback": {...}
    }
  ],
  "final_decision": {...},
  "usage_stats": [...],
  "total_usage": {...}
}
```

## 🛠️ Advanced Features

### Context Sharing Strategies

1. **Independent**: Each expert works in isolation
2. **Previous Answers**: Experts see their own previous responses
3. **Cross-Talk**: Experts can see all other experts' responses and share search context

### Adaptive Evidence Gathering

- **Decompose RAG**: Experts break down questions into specialized sub-queries
- **Few-Shot Learning**: Use example questions to guide expert reasoning
- **Adaptive Search**: Dynamic determination of when additional evidence is needed

### Expert Profile Diversity

Each expert gets a unique profile including:
- Professional job title
- Detailed work experience
- Educational background
- Core specialties
- Unique research focus

## 🔬 Research Applications

MedAgents-2 is designed for:

- **Medical Education**: Training and assessment
- **Clinical Decision Support**: Complex case analysis
- **Medical Research**: Hypothesis generation and validation
- **Quality Assurance**: Medical knowledge validation

## 📝 Example Questions

The system handles various medical question types:

- **Clinical Scenarios**: Patient case analysis
- **Ethical Dilemmas**: Medical ethics and decision-making
- **Diagnostic Challenges**: Complex diagnostic reasoning
- **Treatment Planning**: Therapeutic decision-making

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on OpenAI's GPT models
- Inspired by medical education and clinical decision support systems
- Designed for evidence-based medical reasoning

---

For more information, see the individual component documentation and configuration files. 

## Ablation Study Parameters

| Parameter                    | Config Location           | Description / Ablation Axis                        |
|------------------------------|--------------------------|----------------------------------------------------|
| model.name                   | model                    | Base LLM model (e.g., gpt-4o-mini, gpt-4)          |
| model.temperature            | model                    | LLM sampling temperature                           |
| model.max_tokens             | model                    | Max tokens per response                            |
| retrieval.retrieve_topk      | retrieval                | # docs to retrieve per query (Retrieval)           |
| retrieval.rerank_topk        | retrieval                | # docs to rerank (Retrieval)                       |
| retrieval.allowed_sources    | retrieval                | Document sources to use (Retrieval)                |
| retrieval.rewrite            | retrieval                | Enable query rewriting (Retrieval)                 |
| retrieval.review             | retrieval                | Enable document review (Retrieval)                 |
| retrieval.similarity_strategy| retrieval                | Evidence reuse/generation (reuse/generate/none)    |
| retrieval.query_similarity_threshold | retrieval        | Similarity threshold for query reuse               |
| retrieval.relevance_threshold| retrieval                | Min relevance score for document inclusion         |
| retrieval.search_history     | retrieval                | Search history mode (none/individual/shared)       |
| orchestration.context_sharing| orchestration            | Expert info sharing (independent/cross_talk/...)   |
| orchestration.orchestrator_model | orchestration         | Model for orchestrator agent                       |
| orchestration.orchestrator_temperature | orchestration  | Temperature for orchestrator agent                 |
| difficulty.easy.num_experts  | difficulty               | # experts for easy questions                       |
| difficulty.medium.num_experts| difficulty               | # experts for medium questions                     |
| difficulty.hard.num_experts  | difficulty               | # experts for hard questions                       |
| difficulty.easy.max_round    | difficulty               | # rounds for easy questions                        |
| difficulty.medium.max_round  | difficulty               | # rounds for medium questions                      |
| difficulty.hard.max_round    | difficulty               | # rounds for hard questions                        |
| difficulty.easy.search_mode   | difficulty     | Search tool usage: required, auto, or none         |
| difficulty.medium.search_mode | difficulty     | ''                                                |
| difficulty.hard.search_mode   | difficulty     | ''                                                | 