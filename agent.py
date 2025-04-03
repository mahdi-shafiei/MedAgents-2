import os
import json
import time
import argparse
import random
from typing import List, Dict, Tuple, Optional, Any
from openai import AzureOpenAI, OpenAI
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pymilvus import MilvusClient
from retriever import MedCPTRetriever, calculate_query_similarity
from constants import FORMAT_INST, SEARCH_TOOL, DECOMPOSE_QUERY_SCHEMA, EXPERT_RESPONSE_SCHEMA, MODERATOR_RESPONSE_SCHEMA
import logging
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored terminal output
colorama.init()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

retrieval_client = MilvusClient(uri=os.getenv("MILVUS_URI"))

llm_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION")
)

def _format_question(question: str, options: Dict[str, str]) -> str:
    text = f"{question}\n\n"
    for choice, option in options.items():
        text += f"({choice}) {option}\n"
    return text

class LLMAgent:
    """A medical expert agent powered by a large language model.

    This agent represents a medical expert in a specific domain and handles interactions with an LLM API to generate responses to medical queries.

    Args:
        domain (str): The medical specialty/domain of the expert.
        system_prompt (str, optional): Custom system prompt. If not provided, uses default domain-based prompt.
        args (argparse.Namespace): Configuration arguments for the LLM.

    Attributes:
        domain (str): The agent's medical specialty/domain.
        system_prompt (str): The system prompt defining the agent's role.
        memory (list): Conversation memory as list of message dicts.
        token_usage (dict): Tracks prompt and completion token usage.
    """
    def __init__(self, domain: str, system_prompt: str = None, args: argparse.Namespace = None):
        self.domain = domain
        self.system_prompt = (
            system_prompt.format(self.domain) if system_prompt 
            else f"You are a medical expert in the domain of {self.domain}."
        )
        self.memory = [{'role': 'system', 'content': self.system_prompt}]
        self.token_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        self.args = args
        self.max_retries = getattr(args, 'max_retries', 5) 

    # COMMENT(jw): Do we need this?
    def __repr__(self):
        return f"LLMAgent(\n\tdomain={self.domain},\n\tsystem_prompt={self.system_prompt}\n)"
        
    def chat(self, input_text: str, return_dict: Dict[str, any] = None, save: bool = False, tools: List[Dict] = None, tool_choice: str = None) -> str:
        """Generates a response to the input text using the LLM in JSON mode if schema is provided.

        Args:
            input_text (str): The input prompt/question.
            return_dict (Dict[str, any], optional): A JSON schema dict specifying the expected response.
            save (bool): Whether to save the interaction in conversation memory.
            tools (List[Dict], optional): A list of tool definitions that the model can use.
            tool_choice (str, optional): The tool choice to use.

        Returns:
            str: The generated response text or parsed JSON output if a schema is provided.
        """
        full_input = (
            f"As a {self.domain} expert, {input_text}"
        )

        messages = self.memory + [{'role': 'user', 'content': full_input}]
        print(f"\n--- MESSAGES [{Fore.CYAN}{self.domain}{Style.RESET_ALL}] ---")
        for idx, msg in enumerate(messages):
            role = msg['role'].upper()
            content_preview = str(msg['content'])[:100] + ('...' if len(str(msg['content'])) > 100 else '')
            print(f"{idx}: [{role}] {content_preview}")
        
        response = self._generate_response(messages, return_dict, tools, tool_choice)
        
        if isinstance(response, dict):
            response_text = json.dumps(response)
        else:
            response_text = response

        print(f"\n--- RESPONSE [{Fore.CYAN}{self.domain}{Style.RESET_ALL}] ---")
        try:
            response_preview = response_text[:min(self.args.splice_length, len(response_text))]
            print(response_preview)
        except:
            print(response_text)

        if save:
            self.memory.extend([
                {'role': 'user', 'content': full_input},
                {'role': 'assistant', 'content': response_text}
            ])

        return response

    def _generate_response(self, messages: List[Dict[str, str]], return_dict: Dict[str, any] = None, tools: List[Dict] = None, tool_choice: str = None) -> str:
        """Makes API calls to generate responses, handling retries and errors.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            return_dict (Dict[str, any], optional): JSON schema for the expected response.
            tools (List[Dict], optional): A list of tool definitions that the model can use.

        Returns:
            str: The generated response text or error message, or a tool response object.
        """
        for attempt in range(self.max_retries):
            try:
                request_params = {
                    "model": self.args.model_name,
                    "messages": messages,
                    "max_tokens": self.args.max_tokens,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "seed": self.args.seed,
                    "presence_penalty": self.args.presence_penalty,
                    "frequency_penalty": self.args.frequency_penalty,
                }
                if tools:
                    request_params["tools"] = tools
                if tool_choice:
                    request_params["tool_choice"] = tool_choice
                if return_dict:
                    request_params["response_format"] = {"type": "json_schema", "json_schema": return_dict}

                response = llm_client.chat.completions.create(**request_params)
                self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
                self.token_usage['completion_tokens'] += response.usage.completion_tokens

                if tools:
                    return response.choices[0].message
                elif return_dict:
                    return json.loads(response.choices[0].message.content)
                else:
                    return response.choices[0].message.content

            except Exception as e:
                if "rate" in str(e).lower() or "exceeded" in str(e).lower():
                    print(f'{Fore.YELLOW}Rate limit exceeded. Retrying...{Style.RESET_ALL}')
                    time.sleep(30)
                    continue
                return f"Error generating response: {e}"

        return "Error: Unable to generate response after multiple attempts due to rate limits."

class BaseUnit(ABC):
    def __init__(self, args: argparse.Namespace = None):
        self.args = args
        self.agents = {}

    @abstractmethod
    def run(self):
        pass

    def calculate_token_usage(self):
        token_usage = {'all': {'prompt_tokens': 0, 'completion_tokens': 0}}
        for agent in self.agents.values():
            token_usage['all']['prompt_tokens'] += agent.token_usage['prompt_tokens']
            token_usage['all']['completion_tokens'] += agent.token_usage['completion_tokens']
            token_usage[agent.domain] = agent.token_usage
        return token_usage

class TriageUnit(BaseUnit):
    """A unit that triages medical questions by classifying them into relevant medical specialties 
    and generating comprehensive expert profiles for each selected specialty.

    This unit uses LLM agents to analyze medical questions and determine not only the most appropriate 
    medical specialties but also to construct detailed expert profiles. Each profile includes a special 
    job title, past experiences, educational background, and core specialties that will serve as the system 
    prompt for that expert. It contains two agents:
    - QuestionClassifier: Analyzes the question text to determine relevant specialties and generate expert profiles.
    - OptionClassifier: Analyzes both question and answer options to determine relevant specialties and generate expert profiles.

    Args:
        args (argparse.Namespace): Arguments containing LLM configuration.
    """
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.agents = {
            'QuestionClassifier': LLMAgent(
                "Question Triage", 
                "You are a medical expert specializing in categorizing medical scenarios into specific areas of medicine and in generating detailed expert profiles.", 
                args
            ),
            'OptionClassifier': LLMAgent(
                "Options Triage", 
                "As a medical expert, you possess the ability to discern the most relevant fields of expertise needed to address a multiple-choice question encapsulating a specific medical context, and to generate comprehensive expert profiles.", 
                args
            )
        }

    def triage_question(self, question: str, choices: Dict[str, str], medical_fields: List[str], num_fields: int = 5, options: str = None) -> Dict[str, Tuple[LLMAgent, float]]:
        """Classifies a medical question into relevant specialties, assigns weights, and generates detailed expert profiles.

        For each specialty, the expert profile must include a special job title along with details of past experience,
        educational background, and key specialties.

        Args:
            question (str): The medical question to classify.
            choices (Dict[str, str]): The answer choices with keys like A, B, C, etc.
            medical_fields (List[str]): List of available medical specialties.
            num_fields (int, optional): Number of specialties to select. Defaults to 5.
            options (str, optional): Additional options text to consider.

        Returns:
            Dict[str, Tuple[LLMAgent, float]]: Dictionary mapping each specialty to a tuple of 
            (LLMAgent initialized with the expert's system prompt, weight).
        """
        # Build a JSON schema for the triage response that now includes expert profiles.
        properties = {}
        required_keys = []
        for i in range(num_fields):
            properties[f"Field {i}"] = {"type": "string"}
            properties[f"Weight {i}"] = {"type": "number"}
            properties[f"JobTitle {i}"] = {"type": "string"}
            properties[f"ExpertProfile {i}"] = {"type": "string"}
            properties[f"ResearchFocus {i}"] = {"type": "string"}  # Added research focus for diversity
            required_keys.extend([f"Field {i}", f"Weight {i}", f"JobTitle {i}", f"ExpertProfile {i}", f"ResearchFocus {i}"])
        triage_schema = {
            "name": "triage_response",
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required_keys,
                "additionalProperties": False
            },
            "strict": True
        }
        
        if options:
            prompt = (
                f"You need to complete the following steps:\n"
                f"1. Carefully review the medical scenario:\n'''{_format_question(question, choices)}'''.\n"
                f"2. Examine the extended options provided:\n'''{options}'''. Understand how these options relate to the scenario.\n"
                f"3. Classify the question into {num_fields} subfields chosen from: {', '.join(medical_fields)}.\n"
                f"4. For each selected field, assign a weight between 0 and 1 (the weights must sum to 1).\n"
                f"5. Additionally, for each field, generate a creative, CV-style expert profile along with a unique job title. The profile should be written like a professional curriculum vitae and include (each line should be a separate paragraph):\n"
                f"   - Name: The name of the expert.\n"
                f"   - Past Experience: Provide a detailed summary of clinical or research experience (e.g., 'BS in Biomedical Sciences from Stanford with 5 years in advanced clinical research').\n"
                f"   - Educational Background: List academic degrees, certifications, and specialized training with specifics such as institution names and honors (e.g., 'MD from Harvard, Fellowship in Cardiology at Mayo Clinic').\n"
                f"   - Core Specialties: Enumerate key areas of expertise with precise and creative descriptions.\n"
                f"6. For each expert, also provide a unique research focus that distinguishes them from other experts in the same field. This should be a specific area they've published on or have special interest in.\n"
            )
            response = self.agents['OptionClassifier'].chat(prompt, return_dict=triage_schema)
        else:
            prompt = (
                f"You need to complete the following steps:\n"
                f"1. Read the scenario carefully:\n'''{_format_question(question, choices)}'''.\n"
                f"2. Classify the question into {num_fields} subfields selected from: {', '.join(medical_fields)}.\n"
                f"3. For each selected field, assign a weight between 0 and 1 (weights must sum to 1).\n"
                f"4. Additionally, for each field, generate a creative, CV-style expert profile along with a unique job title. The profile should be written like a professional curriculum vitae and include (each line should be a separate paragraph):\n"
                f"   - Name: The name of the expert.\n"
                f"   - Past Experience: Provide a detailed summary of clinical or research experience (e.g., 'BS in Biomedical Sciences from Stanford with 5 years in advanced clinical research').\n"
                f"   - Educational Background: List academic degrees, supervisors, certifications, and specialized training with specifics such as institution names and honors (e.g., 'MD from Harvard, Fellowship in Cardiology at Mayo Clinic').\n"
                f"   - Core Specialties: Enumerate key areas of expertise with precise and creative descriptions.\n"
                f"5. For each expert, also provide a unique research focus that distinguishes them from other experts in the same field. This should be a specific area they've published on or have special interest in.\n"
            )
            response = self.agents['QuestionClassifier'].chat(prompt, return_dict=triage_schema)
        
        specialty_list = [response[f"Field {i}"] for i in range(num_fields)]
        weights = [float(response[f"Weight {i}"]) for i in range(num_fields)]
        job_titles = [response[f"JobTitle {i}"] for i in range(num_fields)]
        expert_profiles = [response[f"ExpertProfile {i}"] for i in range(num_fields)]
        research_focuses = [response[f"ResearchFocus {i}"] for i in range(num_fields)]
            
        expert_list = {
            specialty: (
                LLMAgent(specialty, f"You are {job_titles[i]} specializing in {specialty}. This is your expert profile:\n{expert_profiles[i]}\nYour unique research focus is: {research_focuses[i]}", args=self.args),
                weights[i]
            )
            for i, specialty in enumerate(specialty_list)
        }
        return expert_list
    
    def run(self, question: str, choices: Dict[str, str], medical_fields: List[str], num_fields: int = 5, options: str = None) -> Dict[str, Tuple[LLMAgent, float]]:
        """Main entry point to run the triage process, which includes both specialty classification and expert profile generation.

        Args:
            question (str): The medical question.
            choices (Dict[str, str]): Answer choices as a dictionary (e.g., {"A": "Option One", ...}).
            medical_fields (List[str]): Available medical specialties.
            num_fields (int): Number of specialties to select.
            options (str, optional): Additional options text.

        Returns:
            Dict[str, Tuple[LLMAgent, float]]: Mapping of each specialty to a tuple of (LLMAgent with expert profile, weight).
        """
        print(f"{Fore.GREEN}Starting triage process to identify relevant medical specialties...{Style.RESET_ALL}")
        expert_list = self.triage_question(question, choices, medical_fields, num_fields, options)
        print(f"{Fore.GREEN}Triage complete. Selected {len(expert_list)} specialties:{Style.RESET_ALL}")
        for specialty, (agent, weight) in expert_list.items():
            print(f"{Fore.CYAN}  • {specialty}{Style.RESET_ALL} (weight: {Fore.YELLOW}{weight:.2f}{Style.RESET_ALL})")
        return expert_list


class ModerationUnit(BaseUnit):
    """A unit that moderates discussions between medical experts and facilitates consensus-building.
    
    This unit contains two agents:
    - Moderator: Summarizes complex discussions and identifies key points.
    - DecisionMaker: Makes evidence-based decisions based on expert input.
    
    It handles the process of weighing expert opinions, making decisions, and summarizing discussions.
    
    Args:
        args: Configuration arguments.
    """
    def __init__(self, args):
        super().__init__(args)
        self.agents = {
            'Moderator': LLMAgent(
                "Moderator",
                system_prompt="You are a skilled and impartial medical debate moderator who can clearly summarize complex discussions and identify key points of agreement and disagreement.",
                args=args
            ),
            'DecisionMaker': LLMAgent(
                "DecisionMaker",
                system_prompt="You are a senior medical expert with experience in consensus-building and evidence-based decision making.",
                args=args
            )
        }
        self.confidence_weights = {
            'high': 1.3,
            'medium': 1.0,
            'low': 0.7
        }

    def make_decision(self, question: str, choices: Dict[str, str], chat_history: Dict[str, Any], agents: Dict[str, Tuple[LLMAgent, float]], final: bool = True) -> Dict[str, Any]:
        """Makes a final decision based on expert discussion using weighted voting.

        Args:
            question (str): The medical question.
            choices (Dict[str, str]): Answer choices (e.g., {"A": "Option1", "B": "Option2", ...}).
            chat_history (Dict[str, Any]): History of expert discussions.
            agents (Dict[str, Tuple[LLMAgent, float]]): Mapping of domain to (agent, weight) tuple.
            final (bool): Whether this is the final decision round.

        Returns:
            Dict[str, Any]: Decision containing final Answer, Justification, Limitations and IsFinal.
        """
        expert_analysis = {}
        for domain, (agent, weight) in agents.items():
            expert_input = chat_history.get(domain, {})
            if not expert_input:
                continue
                
            answer = expert_input.get('answer', '')
            justification = expert_input.get('justification', '')
            confidence = expert_input.get('confidence', 'low')
            expert_analysis[domain] = {
                'answer': answer,
                'justification': justification,
                'confidence': confidence,
                'weight': weight
            }
        vote_results = {key: 0.0 for key in choices.keys()}
        
        for domain, analysis in expert_analysis.items():
            vote_weight = analysis['weight']
            conf = analysis['confidence'].lower()
            vote_weight *= self.confidence_weights.get(conf, 1.0)
            ans = analysis['answer']
            if ans in choices.keys():
                vote_results[ans] += vote_weight
        total_votes = sum(vote_results.values())
        consensus_status = "No consensus reached"
        
        if total_votes > 0:
            vote_shares = {k: v/total_votes for k, v in vote_results.items()}
            winning_choice = max(vote_shares.items(), key=lambda x: x[1])
            if winning_choice[1] > 0.8:
                conf_final = "high"
                consensus_status = "Strong consensus"
            elif winning_choice[1] > 0.6:
                conf_final = "medium"
                consensus_status = "Moderate consensus"
            else:
                conf_final = "low"
                consensus_status = "Weak consensus"
        else:
            winning_choice = (list(choices.keys())[0], 0)
            conf_final = "low"
            consensus_status = "No valid votes recorded"
        decision_prompt = (
            f"Question: {_format_question(question, choices)}\n\n"
            f"Expert Analysis Summary:\n"
        )
        for domain, analysis in expert_analysis.items():
            decision_prompt += f"• {domain} Expert: Answer {analysis['answer']} ({analysis['confidence']} confidence)\n"
        decision_prompt += f"\nVote Distribution: {', '.join([f'{k}: {v:.2f}' for k, v in vote_results.items()])}\n"
        decision_prompt += f"Consensus Status: {consensus_status}\n"
        decision_prompt += f"Preliminary Answer: {winning_choice[0]} (Confidence: {conf_final})\n\n"
        if final:
            decision_prompt += "This is the final decision round. Provide your definitive answer with justification and limitations."
        else:
            decision_prompt += "This is not the final round. Provide an interim decision and note what additional information would help reach consensus."
        MODERATOR_RESPONSE_SCHEMA['schema']['properties']['answer']['enum'] = list(choices.keys())
        print(f"\n{Fore.MAGENTA}Decision Maker analyzing expert opinions...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Vote Distribution: {', '.join([f'{k}: {v:.2f}' for k, v in vote_results.items()])}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Consensus Status: {consensus_status}{Style.RESET_ALL}")
        
        return self.agents['DecisionMaker'].chat(decision_prompt, return_dict=MODERATOR_RESPONSE_SCHEMA, save=True)

    def run(self, question: str, choices: Dict[str, str], chat_history: Dict[str, Any], agents: Dict[str, Tuple[LLMAgent, float]], final: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Runs a round of discussion between experts and makes a decision.

        Args:
            question (str): The medical question.
            choices (Dict[str, str]): Answer choices.
            chat_history (Dict[str, Any]): History of expert discussions.
            agents (Dict[str, Tuple[LLMAgent, float]]): Mapping of domains to (agent, weight) tuples.
            final (bool): Whether this is the final decision round.

        Returns:
            Tuple[str, Dict[str, Any]]: A summary of the discussion and the final decision.
        """
        print(f"\n{Fore.MAGENTA}Moderator starting consensus-building process...{Style.RESET_ALL}")
        response = self.make_decision(question, choices, chat_history, agents, final)
        answer_final = response['answer'].strip().upper()
        if answer_final not in choices.keys():
            answer_final = list(choices.keys())[0]
        response['answer'] = answer_final
        print(f"{Fore.GREEN}Decision: {Fore.WHITE}{answer_final}{Style.RESET_ALL} - {Fore.GREEN}Final: {Fore.WHITE}{response['isFinal']}{Style.RESET_ALL}")
        summary = self.summarize_discussion(chat_history, response)
        return summary, response

    def summarize_discussion(self, expert_solutions: Dict[str, Any], decision: Dict[str, Any]) -> str:
        """Creates a concise summary of the expert discussion and decision.

        Args:
            expert_solutions (Dict[str, Any]): Mapping of expert solutions.
            decision (Dict[str, Any]): Final decision details.

        Returns:
            str: A concise summary of the discussion.
        """
        summarize_prompt = (
            "Create a concise summary of this medical expert discussion. Focus on key points only:\n\n"
            "Expert Opinions:\n"
        )
        
        for domain, solution in expert_solutions.items():
            answer = solution.get('answer', '')
            key_point = solution.get('justification', '')[:100] + "..." if len(solution.get('justification', '')) > 100 else solution.get('justification', '')
            summarize_prompt += f"• {domain}: Answer {answer} - {key_point}\n"
        summarize_prompt += (
            f"\nFinal Decision: {decision.get('answer', '')}\n"
            f"Key Justification: {decision.get('justification', '')}\n"
            f"Limitations: {decision.get('limitations', '')}\n\n"
            "Please provide a 3-5 sentence summary that captures the essence of the discussion, "
            "any major points of agreement/disagreement, and the rationale for the final decision."
        )
        return self.agents['Moderator'].chat(summarize_prompt, save=False)


class SearchUnit(BaseUnit):
    """A unit that handles document retrieval and evaluation for medical questions.

    This unit manages the process of retrieving relevant medical documents, rewriting queries,
    and evaluating document relevance. It contains two agents:
    - QueryRewriter: Rewrites queries to improve search.
    - DocumentEvaluator: Evaluates relevance of retrieved documents.

    Args:
        args: Configuration arguments.
        retriever: Document retrieval system.
        device: Compute device to use.
    """
    def __init__(self, args=None, retriever=None, device=None):
        super().__init__(args)
        self.agents = {
            'QueryRewriter': LLMAgent("Query Rewriter", 
                "Write a medical passage that can help answer the given query. Include key terminology for the answer.", 
                args),
            'DocumentEvaluator': LLMAgent("Document Evaluator", 
                "You are an expert at evaluating a document's usefulness in answering a specific query, not just its relevance. Focus on whether the content truly helps solve the problem posed by the query.", 
                args)
        }
        self.retriever = retriever
        self.device = device
        for attr in ['retrieve_topk', 'rerank_topk', 'allowed_sources']:
            setattr(self, attr, getattr(args, attr, None))
        self.query_cache = []   # query, embedding, documents

    def retrieve_query(self, agent: LLMAgent, query: str) -> List[str]:
        """Retrieves relevant documents for a given question with domain-specific filtering.

        Args:
            agent (LLMAgent): The agent requesting the documents.
            query (str): The medical question.

        Returns:
            List[str]: List of retrieved document texts.
        """
        if agent.domain:
            query += f" Specifically considering aspects related to {agent.domain}."
        
        retrieved_docs = self.retriever.retrieve_filtered_sources(
            query,
            retrieval_client,
            allowed_sources=self.allowed_sources,
            topk=self.retrieve_topk
        )
        reranked_docs = self.retriever.rerank(
            query,
            retrieved_docs,
        )
        docs = reranked_docs[:self.rerank_topk]
        return list(dict.fromkeys(docs))
    
    def review_documents(self, agent: LLMAgent, query: str, documents) -> List[str]:
        """Reviews retrieved documents for relevance with domain-specific criteria.

        Args:
            agent (LLMAgent): The agent requesting the review.
            query (str): The medical question.
            documents (List[str]): List of retrieved documents.

        Returns:
            List[str]: Documents deemed relevant.
        """
        domain_specific_criteria = ""
        if agent.domain:
            domain_specific_criteria = f" Pay special attention to information relevant to {agent.domain}."
            
        review_prompt = (
            "Evaluate the relevance of the following document to the given query\n"
            "Instructions:\n"
            "1. Assess the document's helpfulness in answering the query." + domain_specific_criteria + "\n"
            "2. Label the document as one of:\n"
            "    - [Fully Helpful]: Contains comprehensive and directly relevant info.\n"
            "    - [Partially Helpful]: Some relevant info but incomplete.\n"
            "    - [Not Helpful]: Contains no useful information.\n"
            "3. Respond only with one label: [Fully Helpful], [Partially Helpful], [Not Helpful].\n"
            "Query: {}\n"
            "Document: {}"
        )
        def evaluate_document(doc, idx):
            response = self.agents['DocumentEvaluator'].chat(
                review_prompt.format(query, doc), save=False
            )
            is_helpful = any(label in response.lower() for label in ["ully", "artially"])
            return (doc, is_helpful, idx)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_document, doc, i) for i, doc in enumerate(documents)]
            results = [future.result() for future in futures]
        results.sort(key=lambda x: x[2])
        reviewed_docs = [doc for doc, is_helpful, _ in results if is_helpful]
        
        print(f"{Fore.BLUE}After evaluation: {len(reviewed_docs)}/{len(documents)} documents deemed helpful{Style.RESET_ALL}")
        return reviewed_docs

    def rewrite_query_pseudodoc(self, agent: LLMAgent, query: str) -> str:
        """Rewrites a query to improve document retrieval using domain-specific focus.

        Args:
            agent (LLMAgent): The agent requesting the rewrite.
            query (str): The medical question.

        Returns:
            str: Rewritten query optimized for retrieval.
        """
        domain_specific_instruction = ""
        if agent.domain:
            domain_specific_instruction = f" Focus particularly on aspects related to {agent.domain} and emphasize terminology specific to this field."
            
        few_shot_prompt = f"..."\
            f"Example:"\
            f""\
            f"Query: A 39-year-old woman presents to the family medicine clinic to be evaluated by her physician for weight gain. She reports feeling fatigued most of the day despite eating a healthy diet and exercising regularly. The patient smokes a half-pack of cigarettes daily and has done so for the last 23 years. She is employed as a phlebotomist by the Red Cross. She has a history of hyperlipidemia for which she takes atorvastatin. She is unaware of her vaccination history, and there is no documented record of her receiving any vaccinations. Her heart rate is 76/min, respiratory rate is 14/min, temperature is 37.3°C (99.1°F), body mass index (BMI) is 33 kg/m2, and blood pressure is 128/78 mm Hg. The patient appears alert and oriented. Lung and heart auscultation are without audible abnormalities. The physician orders a thyroid panel to determine if that patient has hypothyroidism. Which of the following recommendations may be appropriate for the patient at this time? A) Hepatitis B vaccination B) Low-dose chest CT C) Hepatitis C vaccination D) Shingles vaccination E) None of the above"\
            f"Passage: against vaccine-preventable diseases. Every visit by an adult to a health-care provider should be an opportunity to provide this protection. Several factors need to be con sidered before any patient is vaccinated. These include the susceptibility of the patient, the risk of exposure to the disease, the risk from the disease, and the benefits and risks from the immunizing agent. Physicians should maintain detailed information about previous vaccina tions received by each individual, including type of vaccination, date of receipt, and adverse events, if any, following vaccination. Information should also include the person's history of vaccine-preventable illnesses, occupation, and lifestyle. Vaccine histories ideally should be based on written documentation to ascertain whether vaccines and toxoids were administered at appropriate ages and at proper intervals. Close attention to factors"\
            f""\
            f"Query: A 23-year-old male presents to his primary care physician after an injury during a rugby game. The patient states that he was tackled and ever since then has had pain in his knee. The patient has tried NSAIDs and ice to no avail. The patient has no past medical history and is currently taking a multivitamin, fish oil, and a whey protein supplement. On physical exam you note a knee that is heavily bruised. It is painful for the patient to bear weight on the knee, and passive motion of the knee elicits some pain. There is laxity at the knee to varus stress. The patient is wondering when he can return to athletics. Which of the following is the most likely diagnosis? A) Medial collateral ligament tear B) Lateral collateral ligament tear C) Anterior cruciate ligament tear D) Posterior cruciate ligament tear E) Meniscal tear F) Patellar dislocation"\
            f"Passage: Diagnosing PCL Injuries: History, Physical Examination, Imaging Studies, Arthroscopic Evaluation. Isolated posterior cruciate ligament (PCL) injuries are uncommon and can be easily missed with physical examination. The purpose of this article is to give an overview of the clinical, diagnostic and arthroscopic evaluation of a PCL injured knee. There are some specific injury mechanisms that can cause a PCL including the dashboard direct anterior blow and hyperflexion mechanisms. During the diagnostic process it is important to distinguish between an isolated or multiligament injury and whether the problem is acute or chronic. Physical examination can be difficult in an acutely injured knee because of pain and swelling, but there are specific functional tests that can indicate a PCL tear. Standard x-ray's and stress views are very useful imaging modalities"\
            f""\
            f"Query: A 45-year-old woman is in a high-speed motor vehicle accident and suffers multiple injuries to her extremities and abdomen. In the field, she was bleeding profusely bleeding and, upon arrival to the emergency department, she is lethargic and unable to speak. Her blood pressure on presentation is 70/40 mmHg. The trauma surgery team recommends emergency exploratory laparotomy. While the patient is in the trauma bay, her husband calls and says that the patient is a Jehovah's witness and that her religion does not permit her to receive a blood transfusion. No advanced directives are available. Which of the following is an appropriate next step? A) Provide transfusions as needed B) Withhold transfusion based on husband's request C) Obtain an ethics consult D) Obtain a court order for transfusion"\
            f"Passage: Legal and ethical issues in safe blood transfusion. This is another D and C Act requirement which is seldom followed, possibly because there are no standard guidelines."\
            f""\
            f"Query: A 4-year-old male is accompanied by his mother to the pediatrician. His mother reports that over the past two weeks, the child has had intermittent low grade fevers and has been more lethargic than usual. The child's past medical history is notable for myelomeningocele complicated by lower extremity weakness as well as bowel and bladder dysfunction. He has been hospitalized multiple times at an outside facility for recurrent urinary tract infections. The child is in the 15th percentile for both height and weight. His temperature is 100.7°F (38.2°C), blood pressure is 115/70 mmHg, pulse is 115/min, and respirations are 20/min. Physical examination is notable for costovertebral angle tenderness that is worse on the right. Which of the following would most likely be found on biopsy of this patient's kidney? A) Mononuclear and eosinophilic infiltrate B) Replacement of renal parenchyma with foamy histiocytes C) Destruction of the proximal tubule and medullary thick ascending limb D) Tubular colloid casts with diffuse lymphoplasmacytic infiltrate"\
            f"Passage: The natural history of urinary infection in adults. The vast majority of otherwise healthy adults with anatomically and functionally normal urinary tracts experience few untoward long-term consequences from symptomatic or asymptomatic UTIs. Effective early treatment of symptomatic infection rapidly curtails bacterial invasion and the resulting inflammatory response. Rarely, uncomplicated acute pyelonephritis causes suppuration and renal scarring. Urinary infections in patients with renal calculi, obstructed urinary tract, neurogenic bladder, or diabetes are frequently much more destructive and have ongoing sequelae. Strategies to treat both the infection and the complications are often necessary to alter this outcome."\
            f"..."\
            f""\
            f"Query: {query} "\
            f"{domain_specific_instruction}"\
            f"Passage:"
        return self.agents['QueryRewriter'].chat(few_shot_prompt, save=False)

    def find_similar_query(self, query_embedding: List[float]) -> Tuple[float, int, List[str]]:
        """Finds if the current query is similar to any previous queries.
        
        Args:
            query_embedding (List[float]): The embedding of the current query.
            
        Returns:
            Tuple[float, int, List[str]]: Maximum similarity score, index of most similar query, and list of previous queries.
        """
        if len(self.query_cache) > 0:
            previous_queries = [item['query'] for item in self.query_cache]
            previous_embeddings = [item['embedding'] for item in self.query_cache]
            max_similarity, max_idx = calculate_query_similarity(query_embedding, previous_embeddings)
            return max_similarity, max_idx, previous_queries
        else:
            return 0, 0, []
    
    def generate_distinct_query(self, agent: LLMAgent, query: str, previous_queries: List[str]) -> str:
        """Generates a new query that is different from previous queries.
        
        Args:
            agent (LLMAgent): The agent requesting the search.
            query (str): The original medical question.
            previous_queries (List[str]): List of previously asked queries.
            
        Returns:
            str: A new, distinct query.
        """
        print(f"{Fore.BLUE}Generating a new query.{Style.RESET_ALL}")
        clarify_prompt = (
            f"Your previous question '{query}' is too similar to questions already asked.\n"
            f"Please generate a completely different question that explores a new aspect of the main problem.\n"
            f"Previous queries include: {', '.join(previous_queries)}\n"
            f"Ensure your new query addresses a unique angle not covered by these queries."
        )
        
        new_response = agent.chat(clarify_prompt, return_dict=DECOMPOSE_QUERY_SCHEMA, save=False)
        return new_response['Query']
    
    def run(self, agent: LLMAgent, query: str, rewrite=False, review=False) -> List[str]:
        """Handles the retrieval of documents based on the query.
        
        This method also handles the case when a generated query is too similar to previous queries 
        using one of three strategies:
        1. Reuse: Reuse the evidence from the most similar query.
        2. Generate: Generate a completely different query.
        3. None: Do nothing and proceed with the original query.

        Args:
            agent (LLMAgent): The agent requesting the search.
            query (str): The medical question.
            rewrite (bool): Whether to rewrite the query.
            review (bool): Whether to review documents.

        Returns:
            List[str]: List of relevant documents.
        """
        query_embedding = self.retriever.encode(query)
        max_similarity, max_idx, previous_queries = self.find_similar_query(query_embedding)
            
        if max_similarity > self.args.query_similarity_threshold:      
            print(f"{Fore.BLUE}Similarity threshold exceeded. Reusing documents from previous query.{Style.RESET_ALL}")
            if self.args.similarity_strategy == 'reuse':
                reused_documents = self.query_cache[max_idx]['documents']
                return reused_documents
            elif self.args.similarity_strategy == 'generate':
                query = self.generate_distinct_query(agent, query, previous_queries)
        
        og_query = query
        if rewrite:
            query = self.rewrite_query_pseudodoc(agent, query)
        documents = self.retrieve_query(agent, query)
        if review:
            documents = self.review_documents(agent, og_query, documents)
        self.query_cache.append({
            'query': og_query,
            'embedding': self.retriever.encode(og_query),
            'documents': documents
        })
        return documents 


class DiscussionUnit(BaseUnit):
    """A unit that facilitates multi-expert discussions on medical questions.
    
    This unit coordinates the debate process between multiple medical experts, manages
    evidence gathering, and facilitates consensus-building through moderated discussion.
    
    Args:
        args: Configuration arguments.
        expert_list: Dictionary mapping expert domains to (agent, weight) tuples.
        search_unit: Unit for retrieving relevant medical information.
        moderation_unit: Unit for moderating discussions and making decisions.
    """
    def __init__(self, args, expert_list, search_unit, moderation_unit):
        super().__init__(args)
        self.agents = expert_list
        self.search_unit = search_unit
        self.moderation_unit = moderation_unit
        
    def specialize_query(self, agent: LLMAgent, query: str, shared_knowledge: List[Dict[str, Any]]) -> str:
        """Generates a specialized inquiry from an expert agent to explore a specific aspect of the main question.
        
        Args:
            agent (LLMAgent): The expert agent generating the query.
            query (str): The main medical question.
            shared_knowledge (List[Dict[str, Any]]): Collective knowledge from previous queries and answers.
            
        Returns:
            str: The specialized inquiry.
        """
        inquiry_prompt = f"Main question to solve: {query}\n"
        if shared_knowledge:
            # Provide richer context from shared knowledge
            knowledge_context = ""
            for item in shared_knowledge:
                knowledge_context += f"{item['domain']} Expert's Question: {item['question']}\n"
                knowledge_context += f"Answer: {item['answer']}\n"
                if 'evidence' in item:
                    knowledge_context += f"Evidence: {item['evidence']}\n"
                knowledge_context += "\n"
            inquiry_prompt += f"Context from previous investigations:\n\n{knowledge_context}\n\n"
        
        inquiry_prompt += (
            "To make further progress, generate a new and distinct sub-question that builds upon what has already been explored, "
            "but targets an uncovered aspect, knowledge gap, or important nuance still relevant to solving the main question.\n\n"
            "You are contributing to an ongoing investigation—do not repeat previous questions, and aim to add new, useful knowledge to the overall discussion.\n"
            "Focus on specificity, novelty, and depth. Each expert should aim to advance the conversation from their unique domain perspective."
        )
        
        response = agent.chat(inquiry_prompt, return_dict=DECOMPOSE_QUERY_SCHEMA, save=False)
        return response['Query'] 

    def gather_evidence(self, query: str, shared_knowledge: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Performs specialized inquiry-based Retrieval-Augmented Generation (RAG) for a medical query before debate.
        
        This method breaks down a complex medical query into more focused sub-queries from each expert's
        perspective, retrieves relevant evidence, and generates specialized knowledge to be shared
        among all experts before the debate begins. This creates a collective knowledge base that
        all experts can draw from during the debate.
        
        Args:
            query (str): The original medical question to be analyzed.
            shared_knowledge (List[Dict[str, Any]]): Collective knowledge from previous queries and answers.
            
        Returns:
            List[Dict[str, Any]]: Updated shared knowledge with new evidence and answers.
        """
        for domain, (agent, _) in self.agents.items():
            specialized_query = self.specialize_query(agent, query, shared_knowledge)
            evidence = self.search_unit.run(agent, specialized_query, rewrite=self.args.rewrite, review=self.args.review)
            evidence = "\n".join(evidence)
            rag_prompt = (
                "The following is a specialized medical inquiry by an expert. Provide a concise yet informative answer based on the relevant evidence and original question.\n\n"
                f"Relevant Evidence:\n{evidence}\n"
                f"Question:\n{specialized_query}\n"
                "Answer:"
            )
            specialized_answer = agent.chat(rag_prompt, save=False)
            
            shared_knowledge.append({
                'domain': agent.domain,
                'question': specialized_query,
                'answer': specialized_answer,
                'evidence': evidence,
            })
        return shared_knowledge

    def find_evidence(self, agent: LLMAgent, query: str, shared_knowledge: List[Dict[str, Any]]) -> str:
        """Performs adaptive RAG by determining if additional information is needed.
        
        Args:
            agent (LLMAgent): The agent requesting evidence.
            query (str): The query to find evidence for.
            shared_knowledge (List[Dict[str, Any]]): Collective knowledge from previous queries.
            
        Returns:
            str: Retrieved evidence as a formatted string.
        """
        if self.args.adaptive_rag == "none":
            return ""

        adaptive_prompt = (
            f"Based on the information you have, determine if you need to search for additional medical information. Please ensure that any search queries are novel and do not duplicate previous information requests.\n"
            f"Question: {query}\n"
        )
        
        retrieval_response = agent.chat(adaptive_prompt, save=False, tools=SEARCH_TOOL, tool_choice=self.args.adaptive_rag)
        retrieved_evidence = ""
        
        if hasattr(retrieval_response, 'tool_calls') and retrieval_response.tool_calls:
            num_tool_calls = len([tc for tc in retrieval_response.tool_calls if tc.function.name == "search_medical_knowledge"])
            evidence_per_query = self.args.rerank_topk // max(1, num_tool_calls)
            
            for tool_call in retrieval_response.tool_calls[:2]:
                if tool_call.function.name == "search_medical_knowledge":
                    tool_args = json.loads(tool_call.function.arguments)
                    search_query = tool_args.get("query", query)
                    rewrite = tool_args.get("options", {}).get("rewrite", self.args.rewrite)
                    review = tool_args.get("options", {}).get("review", self.args.review)
                    new_evidence = self.search_unit.run(agent, search_query, rewrite=rewrite, review=review)
                    new_evidence = new_evidence[:evidence_per_query]

                    shared_knowledge.append({
                        'domain': agent.domain,
                        'question': search_query,
                        'evidence': new_evidence,
                        'answer': ''
                    })
                    
                    if retrieved_evidence:
                        retrieved_evidence += "\n\n"
                    retrieved_evidence += f"- Query: {search_query}\n- Retrieved Evidence:\n" + "\n".join(new_evidence)
            
        return retrieved_evidence

    def debate(self, domain: str, question: str, choices: Dict[str, str], round_num: int, summary: str, shared_knowledge: List[Dict[str, Any]], save: bool) -> Dict[str, Any]:
        """Gets an expert's response for a debate round using JSON mode.

        Args:
            domain (str): Expert domain.
            question (str): The medical question.
            choices (Dict[str, str]): Answer choices.
            round_num (int): Current debate round number.
            summary (str): Summary of previous rounds.
            shared_knowledge (List[Dict[str, Any]]): Collective knowledge from previous queries and answers.
            save (bool): Whether to save agent memory.

        Returns:
            Dict[str, Any]: Expert's response parsed via JSON schema.
        """
        agent, weight = self.agents[domain]
        EXPERT_RESPONSE_SCHEMA['schema']['properties']['answer']['enum'] = list(choices.keys())
        
        if round_num == 0:
            prompt = f"Please analyze the following multiple-choice medical question using your expertise. Provide a systematic analysis and select the most appropriate answer from the given options. Additional reference information is provided to support your analysis.\n\n"
            prompt += f"Question: {_format_question(question, choices)}\n\n"
            
            if self.args.gather_knowledge:
                prompt += "Collective Expert Knowledge:\n"
                for item in shared_knowledge:
                    prompt += f"- Domain: {item['domain']}\n  - Question: {item['question']}\n  - Answer: {item['answer']}\n"
                    if 'evidence' in item and item['evidence']:
                        prompt += f"  - Evidence: {item['evidence']}\n"
                    prompt += "\n"
        else:
            prompt = f"Based on the ongoing expert discussion and newly available information, please reassess your analysis of the medical question below.\n\n"
            prompt += f"Question: {_format_question(question, choices)}\n\n"
            prompt += f"Previous Discussion Summary:\n{summary}\n\n"
        
        evidence = self.find_evidence(agent, prompt, shared_knowledge)
        if evidence:
            prompt += f"Additional Evidence: {evidence}\n\n"
        
        prompt += "Please provide a methodical analysis with your assessment."
        return agent.chat(prompt, return_dict=EXPERT_RESPONSE_SCHEMA, save=save)

    def run(self, question: str, choices: Dict[str, str], max_round: int = 5) -> List[Any]:
        """Main entry point to run the discussion.
        
        Args:
            question (str): The medical question.
            choices (Dict[str, str]): Answer choices.
            max_round (int): Maximum number of debate rounds.

        Returns:
            List[Any]: List of expert responses.
        """
        query = _format_question(question, choices)
        shared_knowledge = []
        
        # Phase 1: Pre-debate knowledge gathering
        # Each expert decomposes the question from their perspective and contributes to shared knowledge
        if self.args.gather_knowledge:
            print(f"\n{Fore.GREEN}Phase 1: Pre-debate knowledge gathering{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'='*50}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Q: {question[:100]}...{Style.RESET_ALL}")
            shared_knowledge = self.gather_evidence(query, shared_knowledge)
            print(f"{Fore.GREEN}Gathered {len(shared_knowledge)} knowledge items{Style.RESET_ALL}")

        # Phase 2: Multi-round debate
        answer_by_turns = []
        chat_history = [{agent: "" for agent in self.agents.keys()} for _ in range(max_round)]
        chat_history_summary = []

        for r in range(max_round):
            print(f"\n{Fore.BLUE}{'='*50}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Phase 2: Round {r+1}{'st' if r == 0 else 'nd' if r == 1 else 'rd' if r == 2 else 'th'} debate{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'='*50}{Style.RESET_ALL}")

            for domain, (agent, weight) in self.agents.items():
                print(f"\n{Fore.CYAN}Expert: {domain} (weight: {weight:.2f}){Style.RESET_ALL}")
                summary = "\n".join([f"{i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} debate summary: {chat}" for i, chat in enumerate(chat_history_summary)])
                response = self.debate(domain, question, choices, r, summary, shared_knowledge, save=self.args.agent_memory)
                chat_history[r][domain] = response
                print(f"{Fore.YELLOW}Answer: {response['answer']} | Confidence: {response['confidence']}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Key point: {response['justification'][:100]}...{Style.RESET_ALL}")
            
            print(f"\n{Fore.MAGENTA}Moderating discussion and building consensus...{Style.RESET_ALL}")
            summary, answer = self.moderation_unit.run(question, choices, chat_history[r], self.agents, final=r == max_round - 1)
            chat_history_summary.append(summary)
            answer_by_turns.append(answer)
            
            print(f"{Fore.MAGENTA}Summary: {summary[:150]}...{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}Current consensus: {answer['answer']} | Justification: {answer['justification'][:100]}... | Limitations: {answer['limitations'][:100]}...{Style.RESET_ALL}")
            
            if answer['isFinal']:
                print(f"\n{Fore.GREEN}Consensus reached after {r+1} rounds. Debate concluded.{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Final answer: {answer['answer']} | Justification: {answer['justification'][:100]}... | Limitations: {answer['limitations'][:100]}...{Style.RESET_ALL}")
                break
            else:
                print(f"\n{Fore.YELLOW}No consensus yet. Moving to next round.{Style.RESET_ALL}")
                
        return answer_by_turns
