import os
import json
import time
from typing import List, Dict, Tuple, Optional
from openai import AzureOpenAI, OpenAI
import argparse
import utils
import warnings
from constants import MEDICAL_SPECIALTIES_GPT_SELECTED, FORMAT_INST
from utils import retrieve, rerank, retrieve_filtered_sources
from abc import ABC, abstractmethod
from pymilvus import MilvusClient

retrieval_client = MilvusClient(uri=os.getenv("MILVUS_URI"))

llm_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"), 
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION")
)

def _format_question(question: str, options: Dict[str, str]) -> str:
    text = f"{question}\n\n"
    for option, choice in options.items():
        text += f"({choice}) {option}\n"
    return text

class LLMAgent:
    """A medical expert agent powered by a large language model.

    This agent represents a medical expert in a specific domain and handles interactions
    with an LLM API to generate responses to medical queries.

    Args:
        domain (str): The medical specialty/domain of the expert
        system_prompt (str, optional): Custom system prompt. If not provided, uses default domain-based prompt
        args (argparse.Namespace): Configuration arguments for the LLM

    Attributes:
        domain (str): The agent's medical specialty/domain
        system_prompt (str): The system prompt defining the agent's role
        history (list): Conversation history as list of message dicts
        token_usage (dict): Tracks prompt and completion token usage
    """

    def __init__(self, domain: str, system_prompt: str = None, args: argparse.Namespace = None):
        self.domain = domain
        self.system_prompt = (
            system_prompt.format(self.domain) if system_prompt 
            else f"You are a medical expert in the domain of {self.domain}."
        )
        self.history = [{'role': 'system', 'content': self.system_prompt}]
        self.token_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        self.args = args
        self.max_retries = getattr(args, 'max_retries', 5)

    def chat(self, input_text: str, return_dict: Dict[str, str] = None, save: bool = True) -> str:
        """Generates a response to the input text using the LLM.

        Args:
            input_text (str): The input prompt/question
            return_dict (Dict[str, str], optional): Expected response format specification
            save (bool): Whether to save the interaction in conversation history

        Returns:
            str: The generated response text
        """
        full_input = (
            f"{input_text}\n\n{FORMAT_INST.format(json.dumps(return_dict, indent=4))}"
            if return_dict else input_text
        )

        messages = self.history + [{'role': 'user', 'content': full_input}]
        response = self._generate_response(messages, return_dict)

        if save:
            self.history.extend([
                {'role': 'user', 'content': full_input},
                {'role': 'assistant', 'content': response}
            ])

        print("\n--------------FULL_INPUT--------------")
        print(full_input)
        print("\n--------------RESPONSE--------------")
        print(response)

        return response

    def _generate_response(self, messages: List[Dict[str, str]], return_dict: Dict[str, str] = None) -> str:
        """Makes API calls to generate responses, handling retries and errors.

        Args:
            messages (List[Dict[str, str]]): The conversation messages
            return_dict (Dict[str, str], optional): Expected response format

        Returns:
            str: The generated response or error message
        """
        for attempt in range(self.max_retries):
            try:
                request_params = {
                    "model": self.args.llm_model,
                    "messages": messages,
                    "max_tokens": self.args.max_tokens,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "seed": self.args.seed,
                    "presence_penalty": self.args.presence_penalty,
                    "frequency_penalty": self.args.frequency_penalty,
                }
                if return_dict:
                    request_params["response_format"] = {"type": "json_object"}

                response = llm_client.chat.completions.create(**request_params)
                self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
                self.token_usage['completion_tokens'] += response.usage.completion_tokens

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
    """A unit that triages medical questions by classifying them into relevant medical specialties.

    This unit uses LLM agents to analyze medical questions and determine the most appropriate medical 
    specialties needed to address them. It contains two agents:
    - QuestionClassifier: Analyzes the question text to determine relevant specialties
    - OptionClassifier: Analyzes both question and answer options to determine relevant specialties

    Args:
        args (argparse.Namespace): Arguments containing LLM configuration
    """
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.agents = {
            'QuestionClassifier': LLMAgent("Question Triage", "You are a medical expert who specializes in categorizing medical scenarios into specific areas of medicine.", args),
            'OptionClassifier': LLMAgent("Options Triage", "As a medical expert, you possess the ability to discern the most relevant fields of expertise needed to address a multiple-choice question encapsulating a specific medical context.", args)
        }

    def triage_question(self, question: str, choices: str, medical_fields: List[str], num_fields: int = 5, options: str = None) -> Tuple[List[LLMAgent], List[float]]:
        """Classifies a medical question into relevant specialties and assigns weights.

        Args:
            question (str): The medical question to classify
            choices (str): The answer choices for the question
            medical_fields (List[str]): List of available medical specialties to choose from
            num_fields (int, optional): Number of specialties to select. Defaults to 5.
            options (str, optional): Additional options text to consider. Defaults to None.

        Returns:
            Tuple[List[LLMAgent], List[float]]: List of LLMAgent experts specialized in the selected medical fields
            and their corresponding weights
        """
        domain_format = {f"Field{i}": "" for i in range(num_fields)}
        weight_format = {f"Weight{i}": 0.0 for i in range(num_fields)}
        format_dict = {**domain_format, **weight_format}
        
        try:
            if options:
                prompt = (
                    f"You need to complete the following steps:\n"
                    f"1. Carefully read the medical scenario presented in the question: '''{_format_question(question, choices)}'''.\n"
                    f"2. The available options are: '''{options}'''. Strive to understand the fundamental connections between the question and the options.\n"
                    f"3. Based on the medical scenario in it, classify the question into {num_fields} different subfields of medicine: {', '.join(medical_fields)}.\n"
                    f"4. Assign a weight between 0 and 1 to each field indicating its relevance (weights should sum to 1).\n"
                )
                response = self.agents['OptionClassifier'].chat(prompt, return_dict=format_dict)
            else:
                prompt = (
                    f"You need to complete the following steps:\n"
                    f"1. Carefully read the medical scenario presented in the question: '''{_format_question(question, choices)}'''.\n"
                    f"2. Based on the medical scenario in it, classify the question into {num_fields} different subfields of medicine: {', '.join(medical_fields)}.\n"
                    f"3. Assign a weight between 0 and 1 to each field indicating its relevance (weights should sum to 1).\n"
                )
                response = self.agents['QuestionClassifier'].chat(prompt, return_dict=format_dict)
            
            specialty_list = [response[f"Field{i}"] for i in range(num_fields)]
            weights = [float(response[f"Weight{i}"]) for i in range(num_fields)]
            
        except (IndexError, ValueError, Exception):
            warnings.warn("Failed to classify question, using default specialties")
            specialty_list = [f"General Medicine_{i+1}" for i in range(num_fields)]
            weights = [1.0/num_fields] * num_fields
            
        expert_list = {specialty: (LLMAgent(specialty, args=self.args), weights[i]) for i, specialty in enumerate(specialty_list)}
        return expert_list
    
    def run(self, question: str, choices: str, medical_fields: List[str], num_fields: int = 5, options: str = None) -> Tuple[List[LLMAgent], List[float]]:
        """Main entry point to run the triage process.

        Args:
            question: The medical question to analyze
            choices: The answer choices
            medical_fields: Available medical specialties
            num_fields: Number of specialties to select
            options: Additional options text

        Returns:
            Tuple[List[LLMAgent], List[float]]: List of LLMAgent experts for the selected specialties
            and their corresponding weights
        """
        return self.triage_question(question, choices, medical_fields, num_fields, options)


class SearchUnit(BaseUnit):
    """A unit that handles document retrieval and evaluation for medical questions.

    This unit manages the process of retrieving relevant medical documents, rewriting queries,
    and evaluating document relevance. It contains two agents:
    - QueryRewriter: Rewrites queries to improve document retrieval
    - DocumentEvaluator: Evaluates relevance of retrieved documents

    Args:
        args: Arguments containing search configuration
        retriever: Document retrieval system
        device: Compute device to use
    """
    def __init__(self, args=None, retriever=None, device=None):
        super().__init__(args)
        self.agents = {
            'QueryRewriter': LLMAgent("Query Rewriter", 
                "Write a medical passage that can help answer the given query. Include key information or terminology for the answer.", 
                args),
            'DocumentEvaluator': LLMAgent("Document Evaluator", 
                "You are an expert in evaluating the relevance of documents to a given query.", 
                args)
        }
        self.retriever = retriever
        self.device = device
        for attr in ['retrieve_topk', 'rerank_topk', 'allowed_sources']:
            setattr(self, attr, getattr(args, attr))
        self.oom_count = 0

    def retrieve_query(self, question, choices) -> List[str]:
        """Retrieves relevant documents for a given question.

        Args:
            question: The medical question
            choices: The answer choices

        Returns:
            List[str]: List of retrieved document texts
        """
        formatted_query = _format_question(question, choices)
        try:
            # First retrieve filtered sources
            retrieved_docs = self.retriever.retrieve_filtered_sources(
                formatted_query,
                retrieval_client,
                self.allowed_sources,
                self.device,
                self.retrieve_topk
            )
            
            # Then rerank them
            reranked_docs = self.retriever.rerank(
                formatted_query,
                retrieved_docs,
                self.device
            )
            
            # Get top k after reranking
            docs = reranked_docs[:self.rerank_topk]
            
        except Exception as e:
            if "memory" in str(e).lower():
                print('retrieve memory error')
                self.oom_count += 1
            return []

        # Remove duplicates while preserving order
        return list(dict.fromkeys(docs))

    def review_documents(self, question, choices, documents) -> List[str]:
        """Reviews retrieved documents for relevance.

        Args:
            question: The medical question
            choices: The answer choices 
            documents: List of documents to review

        Returns:
            List[str]: List of documents deemed relevant
        """
        review_prompt = (
            "Evaluate the relevance of the following document to the given query\n"
            "Instructions:\n"
            "1. Assess the document's helpfulness in answering the query.\n"
            "2. Label the document as one of the following:\n"
            "    - [Fully Helpful]: Contains comprehensive and relevant information directly addressing the query.\n"
            "    - [Partially Helpful]: Contains some relevant information but lacks full coverage or precision.\n"
            "    - [Not Helpful]: Contains no useful information for the query.\n"
            "3. Only respond with one of the labels: [Fully Helpful], [Partially Helpful], [Not Helpful].\n"
            "Query: {}\n"
            "Document: {}"
        )
        
        reviewed_docs = []
        formatted_query = _format_question(question, choices)
        
        for doc in documents:
            try:
                response = self.agents['DocumentEvaluator'].chat(
                    review_prompt.format(formatted_query, doc), save=False      # COMMENT(dainiu): we don't need to save history here
                )
                if any(label in response.lower() for label in ["ully", "artially"]):
                    reviewed_docs.append(doc)
            except Exception:
                continue

        return reviewed_docs

    def rewrite_query_pseudodoc(self, question, choices):
        """Rewrites a query to improve document retrieval.

        Args:
            question: The medical question
            choices: The answer choices

        Returns:
            str: Rewritten query optimized for retrieval
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

    def run(self, question, choices, rewrite=False, review=False):
        """Main entry point to run the search process.

        Args:
            question: The medical question
            choices: The answer choices
            rewrite (bool): Whether to rewrite the query
            review (bool): Whether to review retrieved documents

        Returns:
            List[str]: List of relevant documents
        """
        og_question = question
        if rewrite:
            question = self.rewrite_query_pseudodoc(question, choices)
        documents = self.retrieve_query(question, choices)
        if review:
            documents = self.review_documents(og_question, choices, documents)
        return documents


class ModerationUnit(BaseUnit):
    def __init__(self, args):
        super().__init__(args)
        self.agents = {
            'Moderator': LLMAgent(
                "Moderator",
                system_prompt = "You are a skilled and impartial moderator overseeing a medical debate.",
                args=args
            ),
            'DecisionMaker': LLMAgent(
                "DecisionMaker",
                system_prompt = "You are a knowledgeable medical expert.",
                args=args
            )
        }

    def make_decision(self, question: str, choices: List[str], chat_history: Dict[str, str], agents: Dict[str, Tuple[str, float]], final: bool = True) -> Dict[str, str]:
        """Makes a final decision based on expert discussion.
        
        Args:
            question: The medical question being discussed
            choices: Available answer choices 
            chat_history: History of expert discussions
            agents: Dictionary mapping domain to (agent, weight) tuple
            final: Whether this is the final decision round

        Returns:
            dict: Decision containing FinalAnswer, Confidence, Justification, Limitations and IsFinal
        """
        # Extract each agent's answer, justification and confidence from chat history
        expert_analysis = {}
        for domain, (_, weight) in agents.items():
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

        # Calculate weighted votes for each choice considering confidence
        vote_results = {choice: 0.0 for choice in choices}
        
        for domain, analysis in expert_analysis.items():
            # Apply evidence and confidence-based weighting
            vote_weight = analysis['weight']
            confidence = analysis['confidence'].lower()
            
            # Adjust weight based on confidence
            if confidence == 'high':
                vote_weight *= 1.2
            elif confidence == 'medium':
                vote_weight *= 1.0
            else:  # low confidence
                vote_weight *= 0.8
                
            justification = analysis['justification'].lower()
            if "research evidence" in justification:
                vote_weight *= 1.1
            if "clinical experience" in justification:
                vote_weight *= 1.1
                
            # Count weighted vote
            answer = analysis['answer']
            if answer in choices:
                vote_results[answer] += vote_weight

        total_votes = sum(vote_results.values())
        missing_expertise = []
        
        if total_votes > 0:
            vote_shares = {k: v/total_votes for k,v in vote_results.items()}
            winning_choice = max(vote_shares.items(), key=lambda x: x[1])
            confidence = "high" if winning_choice[1] > 0.7 else "medium" if winning_choice[1] > 0.5 else "low"

            if confidence == "low":
                missing_expertise.append("Need more expert opinions due to low confidence")
        else:
            winning_choice = (list(choices)[0], 0)
            confidence = "low"
            missing_expertise.append("No valid votes recorded")

        # Have decision maker validate and explain
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
        decision_prompt += f"Preliminary Answer: {winning_choice[0]} (Confidence: {confidence})\n\n"
        if missing_expertise:
            decision_prompt += f"Missing Expertise: {', '.join(missing_expertise)}\n\n"
        decision_prompt += "Please validate this decision and provide your final analysis."

        if final:
            decision_prompt += "This is the final decision round. You should provide a final answer."
        else:
            decision_prompt += "This is not the final decision round. But you can provide a final answer if you think it is necessary."

        return self.agents['DecisionMaker'].chat(
            decision_prompt,
            {
                'Answer': f'Select from {" ".join([f"({choice})" for choice in choices])}',
                'Justification': 'Detailed reasoning for the decision',
                'Limitations': 'Key uncertainties or limitations in the analysis',
                'IsFinal': 'true if decision is final, false if more expertise needed'
            },
            save=True,
        )

    def run(self, question: str, choices: str, chat_history: Dict[str, Dict[str, str]], agents: Dict[str, Tuple[str, float]], final: bool = False) -> Tuple[str, str]:
        """Runs a round of discussion between experts and makes a decision.
        
        Args:
            question (str): The medical question being discussed
            choices (list): The answer choices
            chat_history (Dict[str, Dict[str, str]]): History of expert discussions with answers and justifications
            agents (Dict[str, Tuple[str, float]]): List of agents

        Returns:
            Tuple[str, str]: Summary of the discussion and the final decision
        """
        response = self.make_decision(question, choices, chat_history, agents, final)
        response['Answer'] = response['Answer'].replace('(', '').replace(')', '')
        summary = self.summarize_discussion(chat_history, response)
        return summary, response

    def summarize_discussion(self, expert_solutions: Dict[str, Dict[str, str]], decision: Dict[str, str]):
        """Summarizes the key points, conclusions and decision making process from the discussion.
        
        Args:
            expert_solutions: Dictionary mapping domain to expert's answer and justification
            decision: The final decision including answer, justification and limitations
        """
        summarize_prompt = (
            "Summarize the following expert solutions and decision making process. Provide a comprehensive report that includes all perspectives:\n\n"
            "1. **Expert Analysis**:\n"
        )
        
        for domain, solution in expert_solutions.items():
            summarize_prompt += f"\n{domain} Expert:\n"
            summarize_prompt += f"Answer: {solution.get('answer', '')}\n"
            summarize_prompt += f"Justification: {solution.get('justification', '')}\n"
            
        summarize_prompt += (
            "\n2. **Decision Process**:\n"
            f"Final Answer: {decision.get('Answer', '')}\n"
            f"Justification: {decision.get('Justification', '')}\n"
            f"Limitations: {decision.get('Limitations', '')}\n"
            "3. **Detailed Analysis**:\n"
            "   - For each proposed answer, analyze:\n"
            "     - Supporting experts and their key arguments\n"
            "     - Evidence strength and potential limitations\n"
            "     - Contradicting viewpoints\n\n"
            "4. **Key Insights**:\n"
            "   - Areas of expert consensus\n"
            "   - Unresolved disagreements\n"
            "   - Suggested follow-up questions\n\n"
            "Respond in the structured format described above."
        )
        
        return self.agents['Moderator'].chat(summarize_prompt, save=False)


# TODO(dainiu): We need better design for the discussion unit.
class DiscussionUnit(BaseUnit):
    def __init__(self, args, expert_list, search_unit, moderation_unit):
        super().__init__(args)
        self.agents = expert_list
        self.search_unit = search_unit
        self.moderation_unit = moderation_unit
        self.q_a_pairs = []

    def decompose_query(self, question, choices, agent, qa_pairs):
        decompose_prompt = f"Main question to solve: {_format_question(question, choices)}\n"
        if qa_pairs:
            qa_context = ""
            for pair in qa_pairs:
                qa_context += (
                    f"{pair['domain']} Expert's Question: {pair['question']},"
                    f"Answer: {pair['answer']}\n\n"
                )
            decompose_prompt += f"Context from previous questions and answers:\n\n{qa_context}\n\n"
        decompose_prompt += (
            "To help answer the main question, generate a new, specific question that focuses on key terms and gaps in the current answers. "
            "This question should explore a unique aspect or a specific detail needed to solve the main question more effectively. "
            "Each expert should generate a distinct question aimed at uncovering more relevant information for the main question."
        )
    
        response = agent.chat(
            decompose_prompt, 
            return_dict={
                'Query': 'A specific question that explores key medical concepts, diagnostic criteria, treatment approaches, or other relevant aspects needed to address the main question'
            },
            save=False  # TODO(dainiu): look into the save flag, does the agent need to look at history when decomposing a question?
        )
        decomposed_query = json.loads(response)['Query']
        return decomposed_query

    # TODO(dainiu): Do we need to run this for each round of debate?
    def decomposed_rag(self, question, choices, rewrite=False, review=False):
        for agent, weight in self.agents.values():
            decomposed_query = self.decompose_query(question, choices, agent, self.q_a_pairs)
            if rewrite == "Both":
                decomposed_documents = (
                    self.search_unit.run(decomposed_query, choices, rewrite=False, review=review)
                    + self.search_unit.run(decomposed_query, choices, rewrite=True, review=review)
                )
            else:
                decomposed_documents = self.search_unit.run(decomposed_query, choices, rewrite=rewrite, review=review)
            joined_documents = "\n".join(decomposed_documents)
            rag_prompt = (
                        "The following is a decomposed medical question made by a medical expert from original question. Provide a brief yet informative answer "
                        "based on the relevant document and the original question provided. Focus on key information "
                        "and ensure that the answer is concise but addresses the main points needed to solve the question.\n\n"
                        f"Relevant document:\n{joined_documents}\n"
                        f"Question:\n{decomposed_query}\n"
                        "Answer:"
                    )
#            rag_prompt = (
#                "The following task involves addressing a decomposed medical question derived from the original question by a medical expert. "
#                "Your objective is to:\n"
#                "1. Analyze the original question to understand its broader context and purpose.\n"
#                "2. Use the decomposed question to focus on a specific aspect or detail of the original question.\n"
#                "3. Integrate insights from the relevant documents to provide a concise and informative answer.\n\n"
#                "**Guidelines for your answer:**\n"
#                "- Ensure the answer directly addresses the decomposed question while aligning it with the context of the original question.\n"
##                "- Use only the information provided in the relevant documents.\n"
##                "- Be concise, focusing on essential details without including unnecessary information.\n\n"
#                f"**Original Question:**\n{question}\n\n"
#                f"**Relevant Documents:**\n{joined_documents}\n\n"
#                f"**Decomposed Question:**\n{decomposed_query}\n\n"
#                "**Your Answer:**"
#            )
            decomposed_answer = agent.chat(rag_prompt, save=False)  # TODO(dainiu): look into the save flag, does the agent need to look at history when decomposing a question?
            self.q_a_pairs.append({
                'domain': agent.domain,
                'weight': weight,
                'question': decomposed_query,
                'answer': decomposed_answer
            })   

    def get_expert_response(self, domain: str, question: str, choices: List[str], og_documents: str, round_num: int, summary: str):
        """Gets expert response for a given debate round.
        
        Args:
            domain: The expert domain
            question: The medical question
            choices: Answer choices 
            og_documents: Original retrieved documents
            round_num: Current debate round number
            summary: Summary of previous debate rounds
            
        Returns:
            str: Expert's response
        """
        agent, weight = self.agents[domain]
        if round_num == 0:
            user_prompt = f"The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer.\n\n"
            user_prompt += f"Here is the relevant document: {og_documents}\n\n"
            user_prompt += f"Here is the question: {_format_question(question, choices)}\n\n"
            user_prompt += "Here are decomposed question and answer pairs:\n"
            for pair in self.q_a_pairs:
                user_prompt += f"- **Domain:** {pair['domain']}\n"
                user_prompt += f"  - **Question:** {pair['question']}\n"
                user_prompt += f"  - **Answer:** {pair['answer']}\n\n"
            response = agent.chat(
                user_prompt, 
                return_dict={
                    'Thought': 'A step-by-step thought process to generate the answer',
                    'Answer': f"Please select one of the options from {choices}",
                    'Confidence': 'Confidence level of the answer, low/medium/high',
                    'Justification': 'Detailed reasoning for the answer'
                },
                save=True
            )
            return json.loads(response)
        else:
            user_prompt = f"Given summaries of solutions and decomposed question and answer pairs to the problem from other medical experts, consider their opinions as additional advice. Please think carefully and provide an updated answer.\n"
            user_prompt += f"Here are summaries of debate from other medical experts:\n{summary}\n"
            user_prompt += f"Here is the question:{_format_question(question, choices)}\n"
            user_prompt += "Here are decomposed question and answer pairs:\n"
            for pair in self.q_a_pairs:
                user_prompt += f"- **Domain:** {pair['domain']}\n"
                user_prompt += f"  - **Question:** {pair['question']}\n"
                user_prompt += f"  - **Answer:** {pair['answer']}\n\n"
            user_prompt += "Please think step-by-step and generate your output."
            response = agent.chat(user_prompt,
                                  return_dict={
                                      'Thought': 'A step-by-step thought process to generate the answer',
                                      'Answer': f"Please select one of the options from {choices}",
                                      'Confidence': 'Confidence level of the answer, low/medium/high',
                                      'Justification': 'Detailed reasoning for the answer'
                                  },
                                  save=True)
            return json.loads(response)

    def run(self, question: str, choices: List[str], max_round: int = 5):
        self.decomposed_rag(question, choices, rewrite=True, review=False)

        answer_by_turns = []
        chat_history = [{agent: "" for agent in self.agents.keys()} for _ in range(max_round)]
        chat_history_summary = []
        og_documents = '\n'.join(self.search_unit.run(question, choices, rewrite=True, review=False))
        for r in range(max_round):
            print("-"*50)
            print(f"{r}th round debate start")
            print("-"*50)
            for domain, (agent, weight) in self.agents.items():
                response = self.get_expert_response(
                    domain,
                    question,
                    choices,
                    og_documents,
                    r,
                    '\n'.join([f"{i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} debate summary: {chat}" for i, chat in enumerate(chat_history_summary)])
                )
                chat_history[r][domain] = response

            summary, answer = self.moderation_unit.run(question, choices, chat_history[r], self.agents, final=r == max_round - 1)
            chat_history_summary.append(summary)
            answer_by_turns.append(answer)
            if answer['IsFinal'] == 'true':
                break
        return answer_by_turns
