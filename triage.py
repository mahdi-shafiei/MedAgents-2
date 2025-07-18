import os
import asyncio
import nest_asyncio
nest_asyncio.apply()

from dataclasses import dataclass, replace
from typing import Dict, List
from dotenv import load_dotenv
import hydra
from pydantic import BaseModel, Field
from omegaconf import DictConfig

from openai import AsyncOpenAI
from agents import Agent, Runner, RunResult, ModelSettings, handoff, set_default_openai_client, set_tracing_disabled, Usage
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from schema import MEDICAL_SPECIALTIES_GPT_SELECTED

# ——————————————————————————————————————————————
# Configuration dataclass for prompts & model
# ——————————————————————————————————————————————
class ExpertProfile(BaseModel):
    name: str = Field(..., description="Name of the expert")
    past_experience: str = Field(..., description="Detailed summary of clinical or research experience")
    educational_background: str = Field(..., description="Academic degrees, supervisors, certifications, and specialized training")
    core_specialties: str = Field(..., description="Key areas of expertise with precise descriptions")

class TriageOutput(BaseModel):
    difficulty: str = Field(
        ..., 
        description="Question difficulty level", 
        pattern="^(easy|medium|hard)$"
    )
    justification: str = Field(..., description="Reasoning for the assigned difficulty level")
    specialties: List[str] = Field(..., description="List of selected medical specialties")
    weights: List[float] = Field(..., description="Weights for each specialty (must sum to 1)")
    job_titles: List[str] = Field(..., description="Unique job titles for each expert")
    expert_profiles: List[ExpertProfile] = Field(..., description="Detailed expert profiles for each specialty")
    research_focuses: List[str] = Field(..., description="Unique research focus areas for each expert")
 
@dataclass
class TriageConfig:
    model_name: str = "gpt-4o"
    temperature: float = 0.1
    difficulty_system_prompt: str = f"{RECOMMENDED_PROMPT_PREFIX}\nYou are a medical education specialist and triage coordinator."
    difficulty_template: str = (
        "You need to complete the following steps:\n\n"
        "STEP 1: Assess the difficulty level of the following medical question:\n\n"
        "{question}\n\n"
        "Please evaluate the question based on these criteria:\n"
        "1. Knowledge depth required (basic facts vs. specialized knowledge)\n"
        "2. Clinical reasoning complexity (straightforward vs. multi-step reasoning)\n"
        "3. Ambiguity level (clear vs. nuanced distinctions between options)\n"
        "4. Specialized terminology or concepts\n"
        "5. Rarity of the medical condition or treatment discussed\n\n"
        "Classify the question as 'easy', 'medium', or 'hard' and provide a brief justification.\n\n"
        "STEP 2: After determining the difficulty level, you MUST transfer to the appropriate triage agent to get the specialty classification and expert profiles. Do not provide the final triage output yourself - you must handoff to the specialized triage agent."
    )
    triage_system_prompt: str = f"{RECOMMENDED_PROMPT_PREFIX}\nYou are a medical question triage expert."
    triage_template: str = (
        "You need to complete the following steps:\n"
        "1. Read the scenario carefully:\n'''{question}'''.\n"
        "2. Classify the question into {n} experts selected from: {fields}.\n"
        "3. For each selected expert, assign a weight between 0 and 1 (weights must sum to 1).\n"
        "4. Additionally, for each expert, generate a creative, CV-style expert profile along with a unique job title. The profile should be written like a professional curriculum vitae and include (each line should be a separate paragraph):\n"
        "   - Name: The name of the expert.\n"
        "   - Past Experience: Provide a detailed summary of clinical or research experience (e.g., 'BS in Biomedical Sciences from Stanford with 5 years in advanced clinical research').\n"
        "   - Educational Background: List academic degrees, supervisors, certifications, and specialized training with specifics such as institution names and honors (e.g., 'MD from Harvard, Fellowship in Cardiology at Mayo Clinic').\n"
        "   - Core Specialties: Enumerate key areas of expertise with precise and creative descriptions.\n"
        "5. For each expert, also provide a unique research focus that distinguishes them from other experts in the same field. This should be a specific area they've published on or have special interest in.\n"
    )

@dataclass
class TriageRunResult:
    """Result from running a triage agent."""
    response: TriageOutput
    usage: Usage

_default_cfg = TriageConfig()

def update_config(**kwargs):
    global _default_cfg
    _default_cfg = replace(_default_cfg, **kwargs)

# ——————————————————————————————————————————————
# Triage agent runner function
# ——————————————————————————————————————————————
async def run_triage_agent(question: str, cfg: DictConfig):
    # ——————————————————————————————————————————————
    # Define TriageAgent variants for each difficulty level
    # ——————————————————————————————————————————————
    def create_triage_agent(difficulty_level: str):
        config = cfg.difficulty[difficulty_level]
        n = config['num_experts']
        fields = MEDICAL_SPECIALTIES_GPT_SELECTED
        
        return Agent(
            name=f"TriageAgent_{difficulty_level.capitalize()}",
            model=cfg.model.name,
            instructions=(
                f"{_default_cfg.triage_system_prompt}\n"
                f"{_default_cfg.triage_template.format(question=question, fields=fields, n=n)}"
            ),
            model_settings=ModelSettings(
                temperature=cfg.model.temperature,
            ),
            output_type=TriageOutput,
        )

    triage_agents = {
        'easy': create_triage_agent('easy'),
        'medium': create_triage_agent('medium'),
        'hard': create_triage_agent('hard')
    }
    
    # ——————————————————————————————————————————————
    # Define TriageAgent that assesses difficulty and coordinates triage
    # ——————————————————————————————————————————————
    triage_agent = Agent(
        name="TriageAgent",
        model=cfg.model.name,
        instructions=(
            f"{_default_cfg.difficulty_system_prompt}\n"
            f"{_default_cfg.difficulty_template.format(question=question)}"
        ),
        model_settings=ModelSettings(
            temperature=cfg.model.temperature,
        ),
        handoffs=[
            handoff(
                agent=triage_agents['easy'],
                tool_name_override="transfer_to_triage_agent_easy",
                tool_description_override="Transfer to the triage agent for easy questions after assessing difficulty as 'easy'."
            ),
            handoff(
                agent=triage_agents['medium'],
                tool_name_override="transfer_to_triage_agent_medium",
                tool_description_override="Transfer to the triage agent for medium questions after assessing difficulty as 'medium'."
            ),
            handoff(
                agent=triage_agents['hard'],
                tool_name_override="transfer_to_triage_agent_hard",
                tool_description_override="Transfer to the triage agent for hard questions after assessing difficulty as 'hard'."
            )
        ]
    )
    
    result = await Runner.run(
        starting_agent=triage_agent,
        input=question,
        context={},
    )
    return TriageRunResult(
        response=result.final_output,
        usage=result.raw_responses[0].usage
    )

def format_question(question: str, options: Dict[str, str]) -> str:
    """Format question and options for all agents (standardized)."""
    text = f"{question}\n\n"
    for key, value in options.items():
        text += f"({key}) {value}\n"
    return text

# ——————————————————————————————————————————————
# Test harness
# ——————————————————————————————————————————————
@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    load_dotenv()
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    set_default_openai_client(client=client, use_for_tracing=False)
    set_tracing_disabled(disabled=True)

    question = "A 64-year-old man presents to the emergency room with a headache and nausea. He reports that he was rocking his grandson to sleep when the symptoms began. He states the pain is constant and is primarily located on his right side. When asked to indicate the area of pain, he says that it surrounds his eye and upper forehead. He had one episode of vomiting. The patient also reports difficulty seeing out of his right eye, which he attributes to excessive tearing. The patient's past medical history is significant for hypertension. His medications include hydrochlorothiazide. His temperature is 98.6°F (37°C), blood pressure is 135/91 mmHg, pulse is 72/min, and respirations are 12/min. The patient's right eye is shown in Figure A. Upon physical examination, the right pupil is minimally responsive to light and the globe feels firm. A right-sided carotid bruit is appreciated. Which of the following is the most appropriate prophylaxis for this patient's condition?"
    options = {
        'A': 'Myocardial infarction',
        'B': 'Angina pectoris',
        'C': 'Pericarditis',
        'D': 'Myocardial ischemia',
        'E': 'None of the above',
    }
    formatted_question = format_question(question, options)
    result = asyncio.run(run_triage_agent(formatted_question, cfg))
    print("Final JSON Output:", result)

if __name__ == "__main__":
    main()
