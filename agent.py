import os
import json
import time
import torch
import argparse
import warnings
from typing import List, Dict, Tuple, Optional, Any
from openai import AzureOpenAI, OpenAI
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pymilvus import MilvusClient
from retriever import MedCPTRetriever
from constants import FORMAT_INST

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

def _calculate_query_similarity(current_embedding, previous_embeddings):
    # Calculate similarities efficiently using batch processing
    current_tensor = torch.tensor(current_embedding).unsqueeze(0)
    previous_tensor = torch.tensor(previous_embeddings)
    
    # Calculate all similarities at once
    similarities = torch.nn.functional.cosine_similarity(
        current_tensor, 
        previous_tensor,
        dim=1 
    )
    # Check if any similarity is above threshold
    max_similarity = torch.max(similarities).item()
    max_idx = torch.argmax(similarities).item()
    return max_similarity, max_idx

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
        
    def chat(self, input_text: str, return_dict: Dict[str, any] = None, save: bool = True, tools: List[Dict] = None) -> str:
        """Generates a response to the input text using the LLM in JSON mode if schema is provided.

        Args:
            input_text (str): The input prompt/question.
            return_dict (Dict[str, any], optional): A JSON schema dict specifying the expected response.
            save (bool): Whether to save the interaction in conversation memory.
            tools (List[Dict], optional): A list of tool definitions that the model can use.

        Returns:
            str: The generated response text or parsed JSON output if a schema is provided.
        """
        full_input = (
            input_text
        )

        messages = self.memory + [{'role': 'user', 'content': full_input}]
        print(f"\n--------------MESSAGES (Agent: {self.domain})--------------")
        for ith, i in enumerate(messages):
            print(f"{ith}: {str(i)[:self.args.splice_length]}")
        response = self._generate_response(messages, return_dict, tools)

#        print("\n--------------FULL_INPUT--------------")
#        print(full_input[:self.args.splice_length])    

        
        if isinstance(response, dict):
            response_text = json.dumps(response)
        else:
            response_text = response

        print(f"\n--------------RESPONSE (Agent: {self.domain})--------------")
        try:    
            print(response_text[:self.args.splice_length])
        except:
            print(response_text)

        if save:
            self.memory.extend([
                {'role': 'user', 'content': full_input},
                {'role': 'assistant', 'content': response_text}
            ])

        return response

    def _generate_response(self, messages: List[Dict[str, str]], return_dict: Dict[str, any] = None, tools: List[Dict] = None) -> str:
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
                    print('Rate limit exceeded. Retrying...')
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
            required_keys.extend([f"Field {i}", f"Weight {i}", f"JobTitle {i}", f"ExpertProfile {i}"])
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
            )
            response = self.agents['QuestionClassifier'].chat(prompt, return_dict=triage_schema)
        
        specialty_list = [response[f"Field {i}"] for i in range(num_fields)]
        weights = [float(response[f"Weight {i}"]) for i in range(num_fields)]
        job_titles = [response[f"JobTitle {i}"] for i in range(num_fields)]
        expert_profiles = [response[f"ExpertProfile {i}"] for i in range(num_fields)]
            
        expert_list = {
            specialty: (
                LLMAgent(specialty, f"You are {job_titles[i]} specializing in {specialty}. This is your expert profile:\n{expert_profiles[i]}", args=self.args),
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
        return self.triage_question(question, choices, medical_fields, num_fields, options)

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
            setattr(self, attr, getattr(args, attr))
        self.oom_count = 0

    def retrieve_query(self, question, choices: Dict[str, str]) -> List[str]:
        """Retrieves relevant documents for a given question.

        Args:
            question: The medical question.
            choices (Dict[str, str]): Answer choices.

        Returns:
            List[str]: List of retrieved document texts.
        """
        formatted_query = _format_question(question, choices)
        try:
            retrieved_docs = self.retriever.retrieve_filtered_sources(
                formatted_query,
                retrieval_client,
                allowed_sources=self.allowed_sources,
                topk=self.retrieve_topk
            )
            
            reranked_docs = self.retriever.rerank(
                formatted_query,
                retrieved_docs,
            )
            
            docs = reranked_docs[:self.rerank_topk]
            
        except Exception as e:
            if "memory" in str(e).lower():
                print('retrieve memory error')
                self.oom_count += 1
            return []

        return list(dict.fromkeys(docs))
    
    def review_documents(self, question, choices: Dict[str, str], documents) -> List[str]:
        """Reviews retrieved documents for relevance.

        Args:
            question: The medical question.
            choices (Dict[str, str]): Answer choices.
            documents: List of retrieved documents.

        Returns:
            List[str]: Documents deemed relevant.
        """
        review_prompt = (
            "Evaluate the relevance of the following document to the given query\n"
            "Instructions:\n"
            "1. Assess the document's helpfulness in answering the query.\n"
            "2. Label the document as one of:\n"
            "    - [Fully Helpful]: Contains comprehensive and directly relevant info.\n"
            "    - [Partially Helpful]: Some relevant info but incomplete.\n"
            "    - [Not Helpful]: Contains no useful information.\n"
            "3. Respond only with one label: [Fully Helpful], [Partially Helpful], [Not Helpful].\n"
            "Query: {}\n"
            "Document: {}"
        )
        
        formatted_query = _format_question(question, choices)
        
        # Process documents in parallel        
        def evaluate_document(doc, idx):
            response = self.agents['DocumentEvaluator'].chat(
                review_prompt.format(formatted_query, doc), save=False
            )
            is_helpful = any(label in response.lower() for label in ["ully", "artially"])
            return (doc, is_helpful, idx)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_document, doc, i) for i, doc in enumerate(documents)]
            results = [future.result() for future in futures]
        
        # Sort results by original index to maintain input order
        results.sort(key=lambda x: x[2])
        
        # Filter only helpful documents
        reviewed_docs = [doc for doc, is_helpful, _ in results if is_helpful]

        return reviewed_docs

    def rewrite_query_pseudodoc(self, question, choices: Dict[str, str]) -> str:
        """Rewrites a query to improve document retrieval using JSON mode.

        Args:
            question: The medical question.
            choices (Dict[str, str]): The answer choices.

        Returns:
            str: Rewritten query optimized for retrieval.
        """
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
            f"Query: {question} {' '.join([key + ') :' + value for key, value in choices.items()])}"\
            f"Passage:"
        return self.agents['QueryRewriter'].chat(few_shot_prompt, save=False)  # COMMENT(dainiu): we don't need to save history here

    def run(self, question, choices: Dict[str, str], rewrite=False, review=False):
        """Main entry point to run the search process.

        Args:
            question: The medical question.
            choices (Dict[str, str]): Answer choices.
            rewrite (bool): Whether to rewrite the query.
            review (bool): Whether to review documents.

        Returns:
            List[str]: List of relevant documents.
        """
        og_question = question
        if rewrite == "True":
            question = self.rewrite_query_pseudodoc(question, choices)
        documents = self.retrieve_query(question, choices)
        if review == "True":
            documents = self.review_documents(og_question, choices, documents)
        return documents

class ModerationUnit(BaseUnit):
    def __init__(self, args):
        super().__init__(args)
        self.agents = {
            'Moderator': LLMAgent(
                "Moderator",
                system_prompt="You are a skilled and impartial moderator overseeing a medical debate.",
                args=args
            ),
            'DecisionMaker': LLMAgent(
                "DecisionMaker",
                system_prompt="You are a knowledgeable medical expert.",
                args=args
            )
        }

    def make_decision(self, question: str, choices: Dict[str, str], chat_history: Dict[str, Any], agents: Dict[str, Tuple[LLMAgent, float]], final: bool = True) -> Dict[str, str]:
        """Makes a final decision based on expert discussion using JSON mode.

        Args:
            question (str): The medical question.
            choices (Dict[str, str]): Answer choices (e.g., {"A": "Option1", "B": "Option2", ...}).
            chat_history (Dict[str, Any]): History of expert discussions.
            agents (Dict[str, Tuple[LLMAgent, float]]): Mapping of domain to (agent, weight) tuple.
            final (bool): Whether this is the final decision round.

        Returns:
            Dict[str, str]: Decision containing final Answer, Justification, Limitations and IsFinal.
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
            
            # Adjust weight based on confidence
            if conf == 'high':
                vote_weight *= 1.2
            elif conf == 'medium':
                vote_weight *= 1.0
            else:
                vote_weight *= 0.8
                
            justification_lower = analysis['justification'].lower()
            if "research evidence" in justification_lower:
                vote_weight *= 1.1
            if "clinical experience" in justification_lower:
                vote_weight *= 1.1
                
            ans = analysis['answer']
            if ans in choices.keys():
                vote_results[ans] += vote_weight

        total_votes = sum(vote_results.values())
        missing_expertise = []
        
        if total_votes > 0:
            vote_shares = {k: v/total_votes for k, v in vote_results.items()}
            winning_choice = max(vote_shares.items(), key=lambda x: x[1])
            conf_final = "high" if winning_choice[1] > 0.7 else "medium" if winning_choice[1] > 0.5 else "low"

            if conf_final == "low":
                missing_expertise.append("Need more expert opinions due to low confidence")
        else:
            winning_choice = (list(choices.keys())[0], 0)
            conf_final = "low"
            missing_expertise.append("No valid votes recorded")

        # Build a decision prompt and define a JSON schema for the expected response.
        decision_prompt = (
            f"Question: {question}\n\n"
            f"Expert Analysis:\n"
        )
        for domain, analysis in expert_analysis.items():
            decision_prompt += f"\n{domain} Expert:\n"
            decision_prompt += f"Answer: {analysis['answer']}\n"
            decision_prompt += f"Justification: {analysis['justification']}\n"
            decision_prompt += f"Confidence: {analysis['confidence']}\n"
        decision_prompt += f"\nVote Distribution:\n{json.dumps(vote_results, indent=2)}\n\n"
        decision_prompt += f"Preliminary Answer: {winning_choice[0]} (Confidence: {conf_final})\n\n"
        if missing_expertise:
            decision_prompt += f"Missing Expertise: {', '.join(missing_expertise)}\n\n"
        decision_prompt += "Please validate this decision and provide your final analysis."
        decision_prompt += " " + ("This is the final decision round. You should provide a final answer." if final else "This is not the final decision round. Provide an answer if needed.")

        decision_schema = {
            "name": "decision_response",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "enum": list(choices.keys())},
                    "justification": {"type": "string"},
                    "limitations": {"type": "string"},
                    "isFinal": {"type": "string", "enum": ["true", "false"]}
                },
                "required": ["answer", "justification", "limitations", "isFinal"],
                "additionalProperties": False
            },
            "strict": True
        }

        return self.agents['DecisionMaker'].chat(
            decision_prompt,
            return_dict=decision_schema,
            save=False,
        )

    def run(self, question: str, choices: Dict[str, str], chat_history: Dict[str, Any], agents: Dict[str, Tuple[LLMAgent, float]], final: bool = False) -> Tuple[str, Dict[str, str]]:
        """Runs a round of discussion between experts and makes a decision using JSON mode.

        Args:
            question (str): The medical question.
            choices (Dict[str, str]): Answer choices.
            chat_history (Dict[str, Any]): History of expert discussions.
            agents (Dict[str, Tuple[LLMAgent, float]]): Mapping of domains to (agent, weight) tuples.
            final (bool): Whether this is the final decision round.

        Returns:
            Tuple[str, Dict[str, str]]: A summary of the discussion and the final decision.
        """
        response = self.make_decision(question, choices, chat_history, agents, final)
        answer_final = response['answer'].strip().upper()
        if answer_final not in choices.keys():
            answer_final = list(choices.keys())[0]
        response['answer'] = answer_final
        summary = self.summarize_discussion(chat_history, response)
        return summary, response

    def summarize_discussion(self, expert_solutions: Dict[str, Any], decision: Dict[str, str]):
        """Summarizes the key points and decision process using JSON mode.

        Args:
            expert_solutions (Dict[str, Any]): Mapping of expert solutions.
            decision (Dict[str, str]): Final decision details.

        Returns:
            str: A comprehensive summary of the discussion.
        """
        summarize_prompt = (
            "Summarize the following expert solutions and decision process. Provide a structured report including:\n\n"
            "1. Expert Analysis:\n"
        )
        for domain, solution in expert_solutions.items():
            summarize_prompt += f"\n{domain} Expert:\n"
            summarize_prompt += f"Answer: {solution.get('answer', '')}\n"
            summarize_prompt += f"Justification: {solution.get('justification', '')}\n"
        summarize_prompt += (
            "\n2. Decision Process:\n"
            f"Final Answer: {decision.get('answer', '')}\n"
            f"Justification: {decision.get('justification', '')}\n"
            f"Limitations: {decision.get('limitations', '')}\n"
            "3. Detailed Analysis:\n"
            "   - Experts' key arguments\n"
            "   - Strengths and limitations\n"
            "   - Contradicting viewpoints\n"
            "4. Key Insights and Follow-up Questions\n\n"
#            "Respond using the structured format in JSON mode."
        )
        
        summary_schema = {
            "name": "discussion_summary",
            "schema": {
                "type": "object",
                "properties": {
                    "Expert_Analysis": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "Answer": {"type": "string"},
                                "Justification": {"type": "string"}
                            },
                            "required": ["Answer", "Justification"]
                        }
                    },
                    "Decision_Process": {
                        "type": "object",
                        "properties": {
                            "Final_Answer": {"type": "string"},
                            "Justification": {"type": "string"},
                            "Limitations": {"type": "string"}
                        },
                        "required": ["Final_Answer", "Justification", "Limitations"]
                    },
                    "Detailed_Analysis": {
                        "type": "object",
                        "properties": {
                            "Experts_Key_Arguments": {"type": "array", "items": {"type": "string"}},
                            "Strengths_and_Limitations": {
                                "type": "object",
                                "properties": {
                                    "Strengths": {"type": "array", "items": {"type": "string"}},
                                    "Limitations": {"type": "array", "items": {"type": "string"}}
                                }
                            },
                            "Contradicting_Viewpoints": {"type": "string"}
                        }
                    },
                    "Key_Insights_and_Follow_up_Questions": {
                        "type": "object",
                        "properties": {
                            "Key_Insights": {"type": "array", "items": {"type": "string"}},
                            "Follow_up_Questions": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "required": ["Expert_Analysis", "Decision_Process", "Detailed_Analysis", "Key_Insights_and_Follow_up_Questions"]
            },
            "strict": True
        }
        
        return self.agents['Moderator'].chat(
            summarize_prompt,
#            return_dict=summary_schema,
            save=False
        )
# TODO(dainiu): We need better design for the discussion unit.
class DiscussionUnit(BaseUnit):
    def __init__(self, args, expert_list, search_unit, moderation_unit):
        super().__init__(args)
        self.agents = expert_list
        self.search_unit = search_unit
        self.moderation_unit = moderation_unit
        self.q_a_pairs = []
        
    def decompose_query(self, question, choices: Dict[str, str], agent: LLMAgent, q_a_pairs: List[Dict[str, str]], save: List[str]) -> Tuple[str, List[float]]:
        """Generates a decomposed query from an expert agent to explore a specific aspect of the main question.
        
        Args:
            question (str): The main medical question.
            choices (Dict[str, str]): Answer choices for the question.
            agent (LLMAgent): The expert agent generating the query.
            q_a_pairs (List[Dict[str, str]]): Previous question-answer pairs.
            save (List[str]): List of stages to save in agent memory.
            
        Returns:
            Tuple[str, List[float]]: The decomposed query and its embedding.
        """
        decompose_prompt = f"Main question to solve: {_format_question(question, choices)}\n"
        if q_a_pairs:
            qa_context = ""
            for pair in q_a_pairs:
                qa_context += f"{pair['domain']} Expert's Question: {pair['question']}, Answer: {pair['answer']}\n\n"
            decompose_prompt += f"Context from previous questions and answers::\n\n{qa_context}\n\n"
        decompose_prompt += (
                "To make further progress, generate a new and distinct sub-question that builds upon what has already been explored, "
                "but targets an uncovered aspect, knowledge gap, or important nuance still relevant to solving the main question.\n\n"
                "You are contributing to an ongoing investigation—do not repeat previous questions, and aim to add new, useful knowledge to the overall discussion.\n"
                "Focus on specificity, novelty, and depth. Each expert should aim to advance the conversation from their unique domain perspective."
#            "To help answer the main question, generate a new, specific question that focuses on key terms and gaps in the current answers. "
#            "This question should explore a unique aspect or a specific detail needed to solve the main question more effectively. "
#            "Each expert should generate a distinct question aimed at uncovering more relevant information for the main question."
        )
        decomposed_query_schema = {
            "name": "decomposed_query_response",
            "schema": {
                "type": "object",
                "properties": {
                    "Query": {"type": "string"}
                },
                "required": ["Query"],
                "additionalProperties": False
            },
            "strict": True
        }
        response = agent.chat(
            decompose_prompt, 
            return_dict=decomposed_query_schema,
            save='decompose_query' in save
        )
        # Encode the current query
        current_query = response['Query']
        current_embedding = self.search_unit.retriever._medcpt_query_encode(current_query)
        
        return current_query, current_embedding
    
    def generate_new_query(self, agent: LLMAgent, current_query: str, previous_questions: List[str], save: List[str]) -> Tuple[str, List[float]]:
        """Generates a new query when the current one is too similar to previous queries.
        
        Args:
            agent (LLMAgent): The expert agent generating the query.
            current_query (str): The currently generated query.
            previous_questions (List[str]): List of previously asked questions.
            save (List[str]): List of stages to save in agent memory.
            
        Returns:
            Tuple[str, List[float]]: The new query and its embedding.
        """
        decomposed_query_schema = {
            "name": "decomposed_query_response",
            "schema": {
                "type": "object",
                "properties": {
                    "Query": {"type": "string"}
                },
                "required": ["Query"],
                "additionalProperties": False
            },
            "strict": True
        }
        
        clarify_prompt = (
            f"Your previous question '{current_query}' is too similar to questions already asked.\n"
            f"Please generate a completely different question that explores a new aspect of the main problem.\n"
            f"Previous questions include: {', '.join(previous_questions)}\n"
            f"Ensure your new question addresses a unique angle not covered by these questions."
        )
        new_response = agent.chat(
            clarify_prompt,
            return_dict=decomposed_query_schema,
            save='decompose_query' in save
        )
        
        # Re-encode the new query
        new_query = new_response['Query']
        new_embedding = self.search_unit.retriever._medcpt_query_encode(new_query)

        if 'decompose_query' in save:
            agent.memory[-1] = {'role': 'assistant', 'content': json.dumps(new_response)}
            
        return new_query, new_embedding
    
    def reuse_documents(self, similar_idx: int) -> List[str]:
        """Reuses documents from a similar previous query.
        
        Args:
            similar_idx (int): Index of the similar query in q_a_pairs.
            
        Returns:
            List[str]: The documents from the similar query.
        """
        # Extract documents from the similar query
        similar_documents = self.q_a_pairs[similar_idx]['documents']
        return similar_documents
    
    def handle_similar_query(self, agent: LLMAgent, current_query: str, current_embedding: List[float], save: List[str]) -> Tuple[str, List[float], Optional[List[str]]]:
        """Handles the case when a generated query is too similar to previous queries.
        
        Args:
            agent (LLMAgent): The expert agent generating the query.
            current_query (str): The currently generated query.
            current_embedding (List[float]): The embedding of the current query.
            save (List[str]): List of stages to save in agent memory.
            
        Returns:
            Tuple[str, List[float], Optional[List[str]]]: The potentially revised query, its embedding, and reused documents if applicable.
        """
        # Check if there are previous queries to compare with
        if not self.q_a_pairs:
            return current_query, current_embedding, None
            
        # Extract previous questions and encode them
        previous_questions = [pair['question'] for pair in self.q_a_pairs]
        previous_embeddings = [pair['embedding'] for pair in self.q_a_pairs]
        
        # Calculate similarity between current query and previous queries
        max_similarity, max_idx = _calculate_query_similarity(current_embedding, previous_embeddings)
        
        if max_similarity > self.args.query_similarity_threshold:            
            if self.args.decomposed_query_strategy == 'reuse':
                # Reuse the documents from the similar query
                reused_documents = self.reuse_documents(max_idx)
                return current_query, current_embedding, reused_documents
            
            elif self.args.decomposed_query_strategy == 'generate':
                # Generate a new query if too similar
                new_query, new_embedding = self.generate_new_query(agent, current_query, previous_questions, save)
                return new_query, new_embedding, None
            
            # If strategy is 'none', we just keep the current query as is
            
        return current_query, current_embedding, None

    def decomposed_rag(self, question, choices, rewrite=False, review=False, save=[]):
        for agent, weight in self.agents.values():
            # Step 1: Generate decomposed query
            decomposed_query, query_embedding = self.decompose_query(question, choices, agent, self.q_a_pairs, save)
            
            # Step 2: Handle similar queries
            final_query, final_embedding, reused_documents = self.handle_similar_query(agent, decomposed_query, query_embedding, save)
            
            # Step 3: Retrieve documents if not reusing
            if reused_documents is None:
                if rewrite == "Both":
                    decomposed_documents = (
                        self.search_unit.run(final_query, choices, rewrite=True, review=review) +
                        self.search_unit.run(final_query, choices, rewrite=False, review=review)
                    )
                else:
                    decomposed_documents = self.search_unit.run(final_query, choices, rewrite=rewrite, review=review)
            else:
                decomposed_documents = reused_documents
            joined_documents = "\n".join(decomposed_documents)
            rag_prompt = (
                "The following is a decomposed medical question by an expert. Provide a concise yet informative answer based on the relevant document and original question.\n\n"
                f"Relevant Document:\n{joined_documents}\n"
                f"Question:\n{decomposed_query}\n"
                "Answer:"
            )
            decomposed_answer = agent.chat(rag_prompt, save='decompose_answer' in save) # each agent save their decomposed question and answer
            self.q_a_pairs.append({
                'domain': agent.domain,
                'weight': weight,
                'question': decomposed_query,
                'embedding': query_embedding,
                'answer': decomposed_answer,
                'documents': joined_documents
            })

    def get_expert_response(self, domain: str, question: str, choices: Dict[str, str], og_documents: str, round_num: int, summary: str, save: List[str]) -> Dict[str, Any]:
        """Gets an expert's response for a debate round using JSON mode.

        Args:
            domain (str): Expert domain.
            question (str): The medical question.
            choices (Dict[str, str]): Answer choices.
            og_documents (str): Original retrieved documents.
            round_num (int): Current debate round number.
            summary (str): Summary of previous rounds.

        Returns:
            Dict[str, Any]: Expert's response parsed via JSON schema.
        """
        agent, weight = self.agents[domain]
        expert_response_schema = {
            "name": "expert_response",
            "schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "answer": {"type": "string", "enum": list(choices.keys())},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "justification": {"type": "string"}
                },
                "required": ["thought", "answer", "confidence", "justification"],
                "additionalProperties": False
            },
            "strict": True
        }
        # Define a tool for the retriever to search for relevant medical information
        tools = [{
            "type": "function",
            "function": {
                "name": "search_medical_knowledge",
                "description": "Search for relevant medical information to help perform the task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The medical question or search query made by the agent"
                        },
                        "options": {
                            "type": "object",
                            "properties": {
                                "rewrite": {
                                    "type": "boolean",
                                    "description": "Whether to rewrite the query for better search results. The query should be rewritten to be more specific and to include more details, or to be more general and to include less details."
                                },
                            },
                            "required": ["rewrite"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["query", "options"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]
        
        if round_num == 0:
            user_prompt = f"The following is a multiple-choice medical question. Solve it step-by-step and choose one option from the given choices.\n\n"
            user_prompt += f"Question: {_format_question(question, choices)}\n\n"
            # Naive RAG
            if self.args.naive_rag == "True":
                user_prompt += f"Relevant Document: {og_documents}\n\n"
            # Decomposed RAG
            if self.args.decomposed_rag == "True":
                user_prompt += "Decomposed Q&A pairs:\n"
                for pair in self.q_a_pairs:
                    user_prompt += f"- Domain: {pair['domain']}\n  - Question: {pair['question']}\n  - Answer: {pair['answer']}\n\n"
            
            response = agent.chat(user_prompt, return_dict=expert_response_schema, save='debate' in save)
            return response
        else:
            # Adaptive RAG
            if self.args.adaptive_rag == "True":
                adaptive_user_prompt = f"Based on the question and previous discussions, determine if you need to search for additional medical information.\n"
                adaptive_user_prompt += f"Question: {_format_question(question, choices)}\n"
                adaptive_user_prompt += f"Debate Summaries:\n{summary}\n"
                retrieval_response = agent.chat(
                    adaptive_user_prompt,
                    save=False,
                    tools=tools
                )
                retrieved_docs = ""
                if hasattr(retrieval_response, 'tool_calls') and retrieval_response.tool_calls:
                    for tool_call in retrieval_response.tool_calls:
                        if tool_call.function.name == "search_medical_knowledge":
                            tool_args = json.loads(tool_call.function.arguments)
                            query = tool_args.get("query", question)
                            query_embedding = self.search_unit.retriever._medcpt_query_encode(query)
                            # Check if the query is similar to any previously decomposed queries
                            if hasattr(self, 'q_a_pairs') and self.q_a_pairs:
                                # Check similarity with previous decomposed queries
                                for pair in self.q_a_pairs:
                                    prev_query = pair['question']
                                    prev_embedding = pair['embedding']
                                    
                                    # Calculate cosine similarity using the similarity function
                                    max_similarity = _calculate_query_similarity(query_embedding, prev_embedding)
                                    
                                    # If similarity exceeds threshold, rewrite the query
                                    if max_similarity > self.args.query_similarity_threshold:
                                        print(f"Found similar query: {prev_query} | Similarity score: {max_similarity}")
                                        previous_questions = [p['question'] for p in self.q_a_pairs]
                                        clarify_prompt = (
                                            f"Your previous question '{query}' is too similar to questions already asked.\n"
                                            f"Please generate a completely different question that explores a new aspect of the main problem.\n"
                                            f"Previous questions include: {', '.join(previous_questions[:3])}\n"
                                            f"Ensure your new question addresses a unique angle not covered by these questions."
                                        )
                                        query_refinement_response = agent.chat(
                                            clarify_prompt,
                                            save=False
                                        )
                                        query = query_refinement_response.content
                                        break
                            rewrite = tool_args.get("options", {}).get("rewrite", self.args.rewrite)
                            new_docs = self.search_unit.run(query, choices, rewrite=rewrite, review=self.args.review)
                            if retrieved_docs:
                                retrieved_docs += "\n\n"
                            retrieved_docs += "\n".join(new_docs)
                response_user_prompt = f"Considering summaries of previous discussions from other experts and the retrieved information, update your answer.\n"
                response_user_prompt += f"Question: {_format_question(question, choices)}\n"
                response_user_prompt += f"Debate Summaries:\n{summary}\n"
                response_user_prompt += f"Retrieved Information:\n{retrieved_docs}\n"
            else:
                response_user_prompt = f"Considering summaries of previous discussions from other experts, update your answer.\n"
                response_user_prompt += f"Question: {_format_question(question, choices)}\n"
                response_user_prompt += f"Debate Summaries:\n{summary}\n"
            response_user_prompt += "Please think step-by-step and provide your output."
            response = agent.chat(response_user_prompt, return_dict=expert_response_schema, save='debate' in save)
            return response

    def run(self, question: str, choices: Dict[str, str], max_round: int = 5) -> List[Any]:
        if self.args.decomposed_rag == "True":
            self.decomposed_rag(question, choices, rewrite=self.args.rewrite, review=self.args.review, save=self.args.agent_memory)
    
        answer_by_turns = []
        chat_history = [{agent: "" for agent in self.agents.keys()} for _ in range(max_round)]
        chat_history_summary = []
        og_documents = "\n".join(self.search_unit.run(question, choices, rewrite=self.args.rewrite, review=self.args.review))
        print(f"og_documents: {og_documents[:self.args.splice_length]}")

        for r in range(max_round):
            print("-"*50)
            print(f"{r+1}{'st' if r == 0 else 'nd' if r == 1 else 'rd' if r == 2 else 'th'} round debate start")
            print("-"*50)
            for domain, (agent, weight) in self.agents.items():
                response = self.get_expert_response(
                    domain,
                    question,
                    choices,
                    og_documents,
                    r,
                    "\n".join([f"{i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} debate summary: {chat}" for i, chat in enumerate(chat_history_summary)]),
                    save=self.args.agent_memory
                )
                chat_history[r][domain] = response

            summary, answer = self.moderation_unit.run(question, choices, chat_history[r], self.agents, final=r == max_round - 1)

            # Convert the entire summary JSON to a string format
#            summary = json.dumps(summary, indent=2)
            chat_history_summary.append(summary)
            answer_by_turns.append(answer)
            if answer['isFinal'] == 'true':
                break
        return answer_by_turns
