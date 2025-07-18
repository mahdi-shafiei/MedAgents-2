"""
Expert Agent for the Medical Agents system using OpenAI Agents SDK.

This module provides an expert agent that can answer medical questions directly
using search tools and expert knowledge. The agent has a profile from the triage agent
and provides comprehensive medical answers.
"""
import os
import asyncio
import nest_asyncio
nest_asyncio.apply()

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI
from agents import Agent, Runner, RunResult, ModelSettings, RunContextWrapper, set_default_openai_client, set_tracing_disabled, Usage
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel, Field

from search_tools import SearchTool, SearchConfig, create_search_tool_instance, get_search_tools

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class ExpertContext:
    """Context object for expert agent runs."""
    expert_profile: Dict[str, Any]
    question: str
    session_id: str = None

@dataclass
class ExpertAgentConfig:
    """Configuration for the expert agent."""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    system_prompt: str = f"{RECOMMENDED_PROMPT_PREFIX}\nYou are a medical expert specialist. You must return response with in 1 round."
    tool_choice: str = "required"  # "required", "optional", "none"

class ExpertResponse(BaseModel):
    thought: str = Field(..., description="The agent's chain-of-thought reasoning")
    answer: str = Field(
        ...,
        description="Selected multiple-choice answer key (A-Z)",
        pattern="^[A-Z]$"
    )
    confidence: str = Field(
        ..., 
        description="Confidence level in the answer", 
        pattern="^(low|medium|high)$"
    )
    justification: str = Field(..., description="Concise reasoning for the selected answer")

@dataclass
class ExpertRunResult:
    """Result from running an expert agent, including search tracking."""
    response: ExpertResponse
    usage: Usage
    search_tool: SearchTool

@dataclass
class ExpertResult:
    profile: Dict[str, Any]
    result: ExpertRunResult
    weight: float
    round_num: int

_default_cfg = ExpertAgentConfig()

def update_expert_agent_config(**kwargs):
    """Update the global expert agent configuration."""
    global _default_cfg
    for key, value in kwargs.items():
        if hasattr(_default_cfg, key):
            setattr(_default_cfg, key, value)

def create_expert_agent(
    cfg: DictConfig,
    search_tool: SearchTool,
    tool_choice: Optional[str] = None
) -> Agent[ExpertContext]:
    """
    Create an expert agent with the specified configuration.
    """
    search_tools = get_search_tools(search_tool=search_tool)
    tool_choice = tool_choice or cfg.retrieval.get('tool_choice', 'required') if hasattr(cfg, 'retrieval') else 'required'
    def get_expert_instructions(context_wrapper: RunContextWrapper[ExpertContext], _: Agent[ExpertContext]) -> str:
        expert_profile = context_wrapper.context.expert_profile
        name = expert_profile.get('name', 'Medical Expert')
        past_experience = expert_profile.get('past_experience', '')
        educational_background = expert_profile.get('educational_background', '')
        core_specialties = expert_profile.get('core_specialties', '')
        research_focus = expert_profile.get('research_focus', '')
        job_title = expert_profile.get('job_title', 'Medical Specialist')
        instructions = (
            f"{_default_cfg.system_prompt}\n\n"
            f"You are {name}, {job_title}.\n\n"
            f"Your Profile:\n"
            f"- Past Experience: {past_experience}\n"
            f"- Educational Background: {educational_background}\n"
            f"- Core Specialties: {core_specialties}\n"
            f"- Research Focus: {research_focus}\n\n"
            f"You are responsible for answering medical questions using your expertise and the search tools available.\n"
            f"Your approach should be:\n"
            f"1. Analyze the medical question carefully from your specialized perspective\n"
            f"2. Use search tools to gather relevant medical information and evidence\n"
            f"3. Apply your expert knowledge to interpret the findings\n"
            f"4. Provide a comprehensive answer with clear reasoning\n"
            f"5. Include relevant clinical considerations and limitations\n\n"
            f"Available search tools:\n"
            f"- search_medical_knowledge: Search for relevant medical information with configurable parameters\n"
            f"- get_previous_queries: Review previous search queries and their results to avoid duplicates\n\n"
            f"Use the search tools to find the most current and relevant medical information to support your analysis."
        )
        if cfg.retrieval.get('search_history', "none") != "none":
            instructions += f"\n\nPrevious Search Queries and Results:\n"
            instructions += f"{search_tool.get_previous_queries()}\n\n"
        return instructions
    agent = Agent[ExpertContext](
        name="ExpertAgent",
        model=cfg.model.name,
        instructions=get_expert_instructions,
        tools=search_tools,
        output_type=ExpertResponse,
        model_settings=ModelSettings(
            temperature=cfg.model.temperature,
            tool_choice=tool_choice if tool_choice != "none" else None
        )
    )
    return agent

def _make_search_tool_config(cfg):
    search_config = SearchConfig()
    if hasattr(cfg, 'retrieval') and hasattr(cfg.retrieval, 'search_tools'):
        search_cfg = cfg.retrieval.search_tools
        search_config.rewrite = search_cfg.get('auto_rewrite', False)
        search_config.review = search_cfg.get('auto_review', False)
        search_config.allowed_sources = search_cfg.get('allowed_sources', ['cpg', 'statpearls', 'recop', 'textbooks'])
        search_config.similarity_strategy = search_cfg.get('similarity_strategy', 'reuse')
        search_config.query_similarity_threshold = search_cfg.get('query_similarity_threshold', 0.85)
    return search_config

async def run_expert_agent(
    question: str,
    expert_profile: Dict[str, Any],
    cfg: Optional[DictConfig] = None,
    session_id: str = None,
    search_tool: Optional[SearchTool] = None
) -> ExpertRunResult:
    """
    Run the expert agent to answer a medical question.
    If search_tool is provided, use it; otherwise, create a new one from config.
    """
    if cfg is None:
        cfg = OmegaConf.create({
            'model': {'name': 'gpt-4o-mini', 'temperature': 0.1},
            'retrieval': {
                'tool_choice': 'required',
                'search_history': "individual",
                'search_tools': {
                    'auto_rewrite': False,
                    'auto_review': False,
                    'allowed_sources': ['cpg', 'statpearls', 'recop', 'textbooks'],
                    'similarity_strategy': 'reuse',
                    'query_similarity_threshold': 0.85,
                    'relevance_threshold': 5,
                    'max_concurrent_searches': 3,
                    'cache_size': 100
                }
            }
        })
    # Create or use provided search tool
    if search_tool is None:
        search_config = _make_search_tool_config(cfg)
        search_tool = create_search_tool_instance(search_config)
    context = ExpertContext(
        expert_profile=expert_profile,
        question=question,
        session_id=session_id
    )
    expert_agent = create_expert_agent(cfg, search_tool)
    input_text = (
        f"Please answer the following medical question using your expertise and the search tools:\n\n"
        f"Question: {question}\n\n"
        f"Provide a comprehensive analysis including:\n"
        f"1. Your thought process and reasoning\n"
        f"2. The most appropriate answer\n"
        f"3. Your confidence level in the answer\n"
        f"4. Clear justification for your response"
    )
    result = await Runner.run(
        starting_agent=expert_agent,
        input=input_text,
        context=context,
        max_turns=3
    )
    if isinstance(result.final_output, ExpertResponse):
        expert_response = result.final_output
    else:
        logger.warning("Expert agent didn't return expected ExpertResponse format")
        expert_response = ExpertResponse(
            thought="Unable to process response properly",
            answer="Unknown",
            confidence="low",
            justification="Error in response processing"
        )
    return ExpertRunResult(
        response=expert_response,
        usage=result.raw_responses[0].usage,
        search_tool=search_tool
    )

def format_question(question: str, options: Dict[str, str]) -> str:
    """Format question and options for all agents (standardized)."""
    text = f"{question}\n\n"
    for key, value in options.items():
        text += f"({key}) {value}\n"
    return text

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    """Test the expert agent with a sample question and expert profile."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    set_default_openai_client(client=client, use_for_tracing=False)
    set_tracing_disabled(disabled=True)
    expert_profile = {
        "name": "Dr. Sarah Chen",
        "job_title": "Senior Cardiologist and Clinical Researcher",
        "past_experience": "MD from Johns Hopkins University with 15 years of clinical practice in cardiology, specializing in interventional cardiology and heart failure management.",
        "educational_background": "MD from Johns Hopkins University School of Medicine, Fellowship in Interventional Cardiology at Mayo Clinic, Board Certified in Cardiovascular Disease",
        "core_specialties": "Interventional cardiology, heart failure, coronary artery disease, cardiac imaging, preventive cardiology",
        "research_focus": "Novel therapeutic approaches for heart failure with preserved ejection fraction"
    }
    question = (
        "A 65-year-old man presents with chest pain that radiates to his left arm. "
        "He has a history of hypertension and diabetes. His ECG shows ST-segment elevation. "
        "What is the most likely diagnosis and what are the immediate management steps?"
    )
    options = {
        'A': 'Myocardial infarction',
        'B': 'Angina pectoris',
        'C': 'Pericarditis',
        'D': 'Myocardial ischemia',
        'E': 'None of the above',
    }
    formatted_question = format_question(question, options)
    print(f"Expert: {expert_profile['name']}")
    print(f"Specialty: {expert_profile['job_title']}")
    print(f"Question: {question}")
    print("\n" + "="*80)
    print("FIRST RUN - WITHOUT PREVIOUS DISCUSSIONS")
    print("="*80)
    result1 = asyncio.run(run_expert_agent(
        formatted_question, 
        expert_profile, 
        cfg, 
        session_id="test_session_001"
    ))
    print(f"\nExpert Response (Run 1):")
    print(f"Answer: {result1.response.answer}")
    print(f"Confidence: {result1.response.confidence}")
    print(f"Thought Process: {result1.response.thought}")
    print(f"Justification: {result1.response.justification}")
    print(f"\nSearch History:")
    print(result1.search_tool.get_previous_queries())
    print("\n" + "="*80)
    print("SECOND RUN - WITH SAME SEARCH TOOL INSTANCE")
    print("="*80)
    follow_up_question = (
        "Given the initial management, what are the key monitoring parameters "
        "and potential complications to watch for in the first 24 hours?"
    )
    options2 = {
        'A': 'Myocardial infarction',
        'B': 'Angina pectoris',
        'C': 'Pericarditis',
        'D': 'Myocardial ischemia',
        'E': 'None of the above',
    }
    formatted_follow_up_question = format_question(follow_up_question, options2)
    print(f"Follow-up Question: {follow_up_question}")
    context2 = ExpertContext(
        expert_profile=expert_profile,
        question=formatted_follow_up_question,
        session_id="test_session_001"
    )
    # Reuse the search tool for the second run
    result2 = asyncio.run(run_expert_agent(
        formatted_follow_up_question,
        expert_profile,
        cfg,
        session_id="test_session_001",
        search_tool=result1.search_tool
    ))
    print(f"\nExpert Response (Run 2):")
    print(f"Answer: {result2.response.answer}")
    print(f"Confidence: {result2.response.confidence}")
    print(f"Thought Process: {result2.response.thought}")
    print(f"Justification: {result2.response.justification}")
    print(f"\nUpdated Search History:")
    print(result2.search_tool.get_previous_queries())

if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf

    # Try to load config from conf/config.yaml if available, else use a minimal config
    config_path = os.path.join(os.path.dirname(__file__), 'conf', 'config.yaml')
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        # Minimal fallback config
        cfg = OmegaConf.create({
            'model': {'name': 'gpt-4o-mini', 'temperature': 0.1},
            'retrieval': {'tool_choice': 'required', 'search_tools': {}},
        })

    expert_profile = {
        "name": "Dr. Sarah Chen",
        "job_title": "Senior Cardiologist and Clinical Researcher",
        "past_experience": "MD from Johns Hopkins University with 15 years of clinical practice in cardiology, specializing in interventional cardiology and heart failure management.",
        "educational_background": "MD from Johns Hopkins University School of Medicine, Fellowship in Interventional Cardiology at Mayo Clinic, Board Certified in Cardiovascular Disease",
        "core_specialties": "Interventional cardiology, heart failure, coronary artery disease, cardiac imaging, preventive cardiology",
        "research_focus": "Novel therapeutic approaches for heart failure with preserved ejection fraction"
    }
    question = (
        "A 65-year-old man presents with chest pain that radiates to his left arm. "
        "He has a history of hypertension and diabetes. His ECG shows ST-segment elevation. "
        "What is the most likely diagnosis and what are the immediate management steps?"
    )
    options = {
        'A': 'Myocardial infarction',
        'B': 'Angina pectoris',
        'C': 'Pericarditis',
        'D': 'Myocardial ischemia',
        'E': 'None of the above',
    }
    formatted_question = format_question(question, options)
    result = asyncio.run(run_expert_agent(
        formatted_question,
        expert_profile,
        cfg,
        session_id="test_session_001"
    ))
    print("\n=== EXPERT AGENT OUTPUT ===")
    print(f"Answer: {result.response.answer}")
    print(f"Confidence: {result.response.confidence}")
    print(f"Thought Process: {result.response.thought}")
    print(f"Justification: {result.response.justification}")
    print(f"\nSearch History:\n{result.search_tool.get_previous_queries()}") 