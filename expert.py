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
from agents import Agent, Runner, RunResult, WebSearchTool, ModelSettings, RunContextWrapper, set_default_openai_client, set_tracing_disabled, Usage
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
    system_prompt: str = f"{RECOMMENDED_PROMPT_PREFIX}\nYou are a medical expert specialist. You must provide your final answer in a single response. Do not ask for clarification or additional information."
    tool_choice: str = "required"  # "required", "optional", "none"

class ExpertResponse(BaseModel):
    thought: str = Field(..., description="The agent's chain-of-thought reasoning")
    answer: str = Field(
        ...,
        description="Selected multiple-choice answer key (A-Z)",
        pattern="^[A-Z]$"
    )
    evidences: List[str] = Field(
        ...,
        description="List of evidences that support the answer"
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
    tool_choice: Optional[str] = None,
    difficulty_level: Optional[str] = None
) -> Agent[ExpertContext]:
    """
    Create an expert agent with the specified configuration.
    """
    # Determine tool_choice and tool availability based on search_mode
    search_mode = None
    if difficulty_level and hasattr(cfg, 'triage') and difficulty_level in cfg.triage:
        search_mode = cfg.triage[difficulty_level].get('search_mode', 'auto')
    # Check cfg.search.search_mode to determine search tool availability
    config_search_mode = cfg.search.get('search_mode', 'both')
    
    if search_mode == 'required':
        tool_choice = 'required'
        if config_search_mode == 'both':
            search_tools = get_search_tools(search_tool=search_tool)
        elif config_search_mode == 'vector':
            search_tools = get_search_tools(search_tool=search_tool)
        elif config_search_mode == 'web':
            search_tools = [WebSearchTool()]
        else:
            search_tools = []
    elif search_mode == 'auto':
        tool_choice = 'auto'
        if config_search_mode == 'both':
            search_tools = get_search_tools(search_tool=search_tool)
        elif config_search_mode == 'vector':
            search_tools = get_search_tools(search_tool=search_tool)
        elif config_search_mode == 'web':
            search_tools = [WebSearchTool()]
        else:
            search_tools = []
    else:  # none
        tool_choice = None
        search_tools = []

    def get_expert_instructions(context_wrapper: RunContextWrapper[ExpertContext], _: Agent[ExpertContext]) -> str:
        expert_profile = context_wrapper.context.expert_profile
        name = expert_profile.get('name', 'Medical Expert')
        past_experience = expert_profile.get('past_experience', '')
        educational_background = expert_profile.get('educational_background', '')
        core_specialties = expert_profile.get('core_specialties', '')
        research_focus = expert_profile.get('research_focus', '')
        job_title = expert_profile.get('job_title', 'Medical Specialist')
        
        # Build tool description based on available tools
        tool_descriptions = []
        if search_tools:
            for tool in search_tools:
                if hasattr(tool, 'name'):
                    if tool.name == 'search_medical_knowledge':
                        tool_descriptions.append("- search_medical_knowledge: Search for relevant medical information with configurable parameters")
                    elif tool.name == 'web_search':
                        tool_descriptions.append("- web_search: Search the web for current medical information")
                else:
                    # For SearchTool instances
                    tool_descriptions.append("- search_medical_knowledge: Search for relevant medical information with configurable parameters")
        
        instructions = (
            f"{_default_cfg.system_prompt}\n\n"
            f"You are {name}, {job_title}.\n\n"
            f"Your Profile:\n"
            f"- Past Experience: {past_experience}\n"
            f"- Educational Background: {educational_background}\n"
            f"- Core Specialties: {core_specialties}\n"
            f"- Research Focus: {research_focus}\n\n"
            f"IMPORTANT: You must provide your complete final answer in a single response. "
            f"Do not ask follow-up questions or request additional information. "
            f"Use the available tools if needed, then provide your definitive answer.\n\n"
            f"You must provide a list of evidences that support your answer. "
            f"CONFIDENCE GUIDELINES: Be honest about your confidence level:\n"
            f"- Use 'high' confidence only when you are very certain based on strong evidence or clear clinical guidelines\n"
            f"- Use 'medium' confidence when you have good reasoning but some uncertainty remains\n"
            f"- Use 'low' confidence when the question is outside your expertise, evidence is limited, or multiple answers seem plausible\n"
            f"- It is better to express appropriate uncertainty than to appear overconfident\n\n"
            f"You are responsible for answering medical questions using your expertise"
        )
        
        if search_tools:
            instructions += f" and the search tools available.\n"
            instructions += (
                f"Your approach should be:\n"
                f"1. Analyze the medical question carefully from your specialized perspective\n"
                f"2. Use search tools to gather relevant medical information and evidence\n"
                f"3. Apply your expert knowledge to interpret the findings\n"
                f"4. Assess your confidence based on the strength of evidence and your expertise in this area\n"
                f"5. Provide a comprehensive answer with clear reasoning in your FINAL response\n"
                f"6. Include relevant clinical considerations and limitations\n\n"
                f"Available search tools:\n"
            )
            instructions += "\n".join(tool_descriptions)
            
            # Add specific guidance for using both tools when available
            if config_search_mode == 'both' and len(search_tools) > 1:
                instructions += f"\n\nSEARCH STRATEGY: You have access to both medical knowledge search and web search tools. "
                instructions += f"For the most comprehensive analysis, consider using BOTH tools:\n"
                instructions += f"- Use search_medical_knowledge first to find established medical knowledge and guidelines\n"
                instructions += f"- Use web_search to find current research, recent developments, or additional perspectives\n"
                instructions += f"- Combining both sources will give you the most complete picture for your analysis"
            
            instructions += f"\n\nUse the search tools to find the most current and relevant medical information to support your analysis. After gathering information, provide your complete final answer immediately with an honest assessment of your confidence."
        else:
            instructions += f".\n"
            instructions += (
                f"Your approach should be:\n"
                f"1. Analyze the medical question carefully from your specialized perspective\n"
                f"2. Apply your expert knowledge and clinical experience\n"
                f"3. Assess your confidence based on your expertise in this specific area\n"
                f"4. Provide a comprehensive answer with clear reasoning\n"
                f"5. Include relevant clinical considerations and limitations\n\n"
                f"Note: No search tools are available for this question, so rely on your medical expertise and provide your complete answer immediately with an honest confidence assessment."
            )
        
        if cfg.search.get('search_history', "none") != "none" and search_tool:
            instructions += f"\n\nPrevious Search Queries and Results:\n"
            instructions += f"{search_tool.get_previous_queries()}\n\n"
        return instructions
    agent = Agent[ExpertContext](
        name="ExpertAgent",
        model=cfg.execution.model.name,
        instructions=get_expert_instructions,
        tools=search_tools,
        output_type=ExpertResponse,
        model_settings=ModelSettings(
            temperature=cfg.execution.model.temperature,
            tool_choice=tool_choice
        )
    )
    return agent

def _make_search_tool_config(cfg):
    search_config = SearchConfig()
    if hasattr(cfg, 'search'):
        search_config.rewrite = getattr(cfg.search, 'rewrite', False)  # config: search.rewrite
        search_config.review = getattr(cfg.search, 'review', False)    # config: search.review
        search_config.allowed_sources = getattr(cfg.search, 'allowed_sources', ['cpg', 'statpearls', 'recop', 'textbooks'])
        search_config.similarity_strategy = getattr(cfg.search, 'similarity_strategy', 'reuse')
        search_config.query_similarity_threshold = getattr(cfg.search, 'query_similarity_threshold', 0.85)
        search_config.retrieve_topk = getattr(cfg.search, 'retrieve_topk', 100)
        search_config.rerank_topk = getattr(cfg.search, 'rerank_topk', 25)
        search_config.cache_size = getattr(cfg.search, 'cache_size', 100)
        search_config.max_concurrent_searches = getattr(cfg.search, 'max_concurrent_searches', 3)
        search_config.relevance_threshold = getattr(cfg.search, 'relevance_threshold', 5)
        search_config.search_history = getattr(cfg.search, 'search_history', 'individual')
    return search_config

async def run_expert_agent(
    question: str,
    expert_profile: Dict[str, Any],
    cfg: Optional[DictConfig] = None,
    session_id: str = None,
    search_tool: Optional[SearchTool] = None,
    difficulty_level: Optional[str] = None
) -> ExpertRunResult:
    """
    Run the expert agent to answer a medical question.
    If search_tool is provided, use it; otherwise, create a new one from config.
    """
    if cfg is None:
        cfg = OmegaConf.create({
            'execution': {'model': {'name': 'gpt-4o-mini', 'temperature': 0.1}},
            'search': {
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
    expert_agent = create_expert_agent(cfg, search_tool, difficulty_level=difficulty_level)
    input_text = (
        f"Please answer the following medical question using your expertise and the search tools:\n\n"
        f"Question: {question}\n\n"
        f"CRITICAL: You must provide your complete final answer in this single interaction. "
        f"Do not ask for clarification or additional information. Use any available search tools if needed, "
        f"then immediately provide your comprehensive analysis including:\n"
        f"1. Your thought process and reasoning\n"
        f"2. The most appropriate answer\n"
        f"3. Your honest confidence level in the answer (be realistic about uncertainty)\n"
        f"4. Clear justification for your response\n\n"
        f"Remember: It is better to express appropriate uncertainty than to appear overconfident. "
        f"Use 'low' confidence when you have doubts, 'medium' when reasonably sure, and 'high' only when very certain.\n\n"
        f"This is your only opportunity to respond - make it complete and definitive."
    )
    for _ in range(5):
        try:
            result = await Runner.run(
                starting_agent=expert_agent,
                input=input_text,
                context=context,
                max_turns=5,
            )
            break
        except Exception:
            pass

    if isinstance(result.final_output, ExpertResponse):
        expert_response = result.final_output
    else:
        logger.warning("Expert agent didn't return expected ExpertResponse format")
        expert_response = ExpertResponse(
            thought="Unable to process response properly",
            answer="Z",
            confidence="low",
            evidences=[],
            justification="Error in response processing"
        )
    total_usage = Usage()
    for raw_response in result.raw_responses:
        total_usage.add(raw_response.usage)
    return ExpertRunResult(
        response=expert_response,
        usage=total_usage,
        search_tool=search_tool
    )

def format_question(question: str, options: Dict[str, str]) -> str:
    """Format question and options for all agents (standardized)."""
    text = f"{question}\n\n"
    for key, value in options.items():
        text += f"({key}) {value}\n"
    return text