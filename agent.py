import os
import json
import time
from typing import List, Dict
from openai import AzureOpenAI, OpenAI
import argparse
import utils
import warnings
from constants import MEDICAL_SPECIALTIES_GPT_SELECTED, FORMAT_INST
from utils import retrieve, rerank, retrieve_filtered_sources
from abc import ABC, abstractmethod

retrieval_client = MilvusClient(uri=os.getenv("MILVUS_URI"))

llm_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION")
)

def _format_question(question, options):
    text = f"{question}\n\n"
    for option, choice in options.items():
        text += f"({choice}) {option}\n"
    return text

class LLMAgent:
    def __init__(self, domain: str, system_prompt: str = None, args: argparse.Namespace = None):
        self.token_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        self.domain = domain
        self.system_prompt = (
            system_prompt.format(self.domain) if system_prompt 
            else f"You are a medical expert in the domain of {self.domain}."
        )
        self.history = [{'role': 'system', 'content': self.system_prompt}]
        self.args = args

    def chat(
        self, 
        input_text: str, 
        return_dict: Dict[str, str] = None, 
        save: bool = False
    ) -> str:
        if return_dict:
            return_instruction = FORMAT_INST.format(json.dumps(return_dict, indent=4))
            full_input = f"{input_text}\n\n{return_instruction}"
        else:
            full_input = input_text

        response = self._generate_response(
            self.history + [{'role': 'user', 'content': full_input}], 
            return_dict
        )

        if save:
            self.history.extend([
                {'role': 'user', 'content': full_input},
                {'role': 'assistant', 'content': response}
            ])

        self._log_interaction(full_input, response)
        return response

    def _generate_response(
        self, 
        messages: List[Dict[str, str]], 
        return_dict: Dict[str, str] = None, 
        max_retries: int = 100
    ) -> str:
        attempt = 0
        while attempt < max_retries:
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
                self._update_token_usage(response)

                return response.choices[0].message.content
            except Exception as e:
                if "rate" in str(e).lower() or "exceeded" in str(e).lower():
                    print('Rate limit exceeded. Retrying...')
                    attempt += 1
                    time.sleep(30)
                else:
                    return f"Error generating response: {e}"
        return "Error: Unable to generate response after multiple attempts due to rate limits."

    def _update_token_usage(self, response):
        self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
        self.token_usage['completion_tokens'] += response.usage.completion_tokens

    def _log_interaction(self, input_text: str, response: str):
        print("\n--------------FULL_INPUT--------------")
        print(input_text)
        print("\n--------------RESPONSE--------------")
        print(response)

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

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
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.agents = {
            'QuestionClassifier': LLMAgent("Question Triage", "You are a medical expert who specializes in categorizing medical scenarios into specific areas of medicine.", args),
            'OptionClassifier': LLMAgent("Options Triage", "As a medical expert, you possess the ability to discern the most relevant fields of expertise needed to address a multiple-choice question encapsulating a specific medical context.", args)
        }

    def run(self, question: str, choices: str, medical_fields: List[str], num_fields: int = 5, options: str = None) -> List[str]:
        domain_format = {f"Field{i}": "" for i in range(num_fields)}
        try:
            if options:
                prompt = (
                    f"You need to complete the following steps:\n"
                    f"1. Carefully read the medical scenario presented in the question: '''{_format_question(question, choices)}'''.\n"
                    f"2. The available options are: '''{options}'''. Strive to understand the fundamental connections between the question and the options.\n"
                    f"3. Based on the medical scenario in it, classify the question into {num_fields} different subfields of medicine: {', '.join(medical_fields)}.\n"
                )
                response = self.agents['OptionClassifier'].chat(prompt, return_dict=domain_format)
            else:
                prompt = (
                    f"You need to complete the following steps:\n"
                    f"1. Carefully read the medical scenario presented in the question: '''{_format_question(question, choices)}'''.\n"
                    f"2. Based on the medical scenario in it, classify the question into {num_fields} different subfields of medicine: {', '.join(medical_fields)}.\n"
                )
                response = self.agents['QuestionClassifier'].chat(prompt, return_dict=domain_format)
            specialty_list = [domain for domain in response.values()]
        except (IndexError, ValueError, Exception):
            warnings.warn("Failed to classify question, using default specialties")
            specialty_list = [f"General Medicine_{i+1}" for i in range(num_fields)]
        expert_list = [LLMAgent(specialty, args=self.args) for specialty in specialty_list]
        return expert_list


class SearchUnit(BaseUnit):
    def __init__(self, args=None, retrieval_client=None, retriever=None, device=None):
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
        self.retrieval_client = retrieval_client
        self.device = device
        for attr in ['retrieve_topk', 'rerank_topk', 'allowed_sources']:
            setattr(self, attr, getattr(args, attr))
        self.oom_count = 0

    def _retrieve_query(self, question, choices) -> List[str]:
        formatted_query = _format_question(question, choices)
        try:
            # First retrieve filtered sources
            retrieved_docs = self.retriever.retrieve_filtered_sources(
                formatted_query,
                self.retrieval_client,
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

    def _review_documents(self, question, choices, documents) -> List[str]:
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
                    review_prompt.format(formatted_query, doc)
                )
                if any(label in response.lower() for label in ["ully", "artially"]):
                    reviewed_docs.append(doc)
            except Exception:
                continue

        return reviewed_docs

    def _rewrite_query_pseudodoc(self, question, choices):
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
        return self.agents['QueryRewriter'].chat(few_shot_prompt)


class ModerationUnit(BaseUnit):
    def __init__(self, args=None):
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

class DiscussionUnit(BaseUnit):
    def __init__(self, expert_list, args=None):
        super().__init__(args)
        self.agents = {agent.domain: agent for agent in expert_list}
        self.search_unit = args.search_unit
        self.moderation_unit = args.moderation_unit
        self.max_iterations = args.llm_debate_max_round
        self.q_a_pairs = []
 
    def _decompose_query(self, question, choices, agent, qa_pairs):
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
    
        response = agent.chat(decompose_prompt, {'Query': 'string'})
        decomposed_query = json.loads(response)['Query']
        return decomposed_query

    def _request_search(self, question, choices, rewrite=False, review=False):
        og_question = question
        if rewrite:
            question = self.search_unit._rewrite_query_pseudodoc(question, choices)
        documents = self.search_unit._retrieve_query(question, choices)
        if review:
            documents = self.search_unit._review_documents(og_question, choices, documents)
        return documents
    
    def _decomposed_rag(self, question, choices, rewrite=False, review=False):
        for idx, agent in self.agents.items():
            decomposed_query = self._decompose_query(question, choices, agent, self.q_a_pairs)
            if rewrite == "Both":
                decomposed_documents = (
                    self._request_search(decomposed_query, choices, rewrite=False, review=review)
                    + self._request_search(decomposed_query, choices, rewrite=True, review=review)
                )
            else:
                decomposed_documents = self._request_search(decomposed_query, choices, rewrite=rewrite, review=review)
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
            decomposed_answer = agent.chat(rag_prompt)
            self.q_a_pairs.append({
                'domain': agent.domain,
                'question': decomposed_query,
                'answer': decomposed_answer
            })   

    def simultaneous_talk_summarizer_v2(self, question, choices, max_round):
        answer_by_turns = []
        chat_history = [[] for _ in range(max_round)]
        chat_history_summary = []
        og_documents = '\n'.join(self._request_search(question, choices, rewrite=True, review=False))
        for r in range(max_round):
            print("-"*50)
            print(f"{r}th round debate start")
            print("-"*50)
            for i, agent in self.agents.items():
                if r == 0:
                    user_prompt = f"The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer.\n\n"
                    user_prompt += f"Here is the relevant document: {og_documents}\n\n"
                    user_prompt += f"Here is the question: {_format_question(question, choices)}\n\n"
                    user_prompt += "Here are decomposed question and answer pairs:\n"
                    for pair in self.q_a_pairs:
                        user_prompt += f"- **Domain:** {pair['domain']}\n"
                        user_prompt += f"  - **Question:** {pair['question']}\n"
                        user_prompt += f"  - **Answer:** {pair['answer']}\n\n"
                    response = agent.chat(user_prompt)
                    chat_history[r].append(f"{agent.domain} Expert's solution{response}")
                else:
                    summaries = '\n'.join([f"{i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} debate summary: {chat}" for i, chat in enumerate(chat_history_summary)])
                    user_prompt = f"Given summaries of solutions and decomposed question and answer pairs to the problem from other medical experts, consider their opinions as additional advice. Please think carefully and provide an updated answer.\n"
                    user_prompt += f"Here are summaries of debate from other medical experts:\n{summaries}\n"
                    user_prompt += f"Here is the question:{_format_question(question, choices)}\n"
                    user_prompt += "Here are decomposed question and answer pairs:\n"
                    for pair in self.q_a_pairs:
                        user_prompt += f"- **Domain:** {pair['domain']}\n"
                        user_prompt += f"  - **Question:** {pair['question']}\n"
                        user_prompt += f"  - **Answer:** {pair['answer']}\n\n"
                    user_prompt += "Please think step-by-step and generate your output."
                    response = agent.chat(user_prompt)
                    chat_history[r].append(response)

            joined_solutions = "\n".join([f"{agent.domain} expert's response: {chat_history[r][i]}" for i, agent in self.agents.items()])
            summarizer_prompt = (
                "Summarize the following expert solutions and provide a comprehensive report that includes all perspectives:\n"
                "1. **Expert Solutions**:\n"
                "   - Expert n: Key points, suggested solution, rationale.\n"
                "   - Repeat as needed for all experts.\n\n"
                "2. **Detailed Analysis**:\n"
                "   - For each proposed answer (A, B, C, D), provide:\n"
                "     - Experts supporting this answer and their rationale.\n"
                "     - Key evidence, strengths, and potential weaknesses for this answer.\n"
                "     - Contradictions or opposing arguments raised by other experts.\n\n"
                "3. **Actionable Insights**:\n"
                "   - Summarize key points where experts agree and why.\n"
                "   - Highlight unresolved disagreements or areas requiring further clarification.\n"
                "   - Propose targeted questions or new angles of inquiry for the next round to address the gaps.\n\n"
                "Here are the expert solutions:\n"
                f"{joined_solutions}\n"
                "Respond in the structured format described above."
            )
            summary = self.moderation_unit.agents['Moderator'].chat(summarizer_prompt)
            chat_history_summary.append(summary)

            summaries = '\n'.join([f"{i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} debate summary: {chat}" for i, chat in enumerate(chat_history_summary)])
            decision_prompt = f"Given the summary of previous debate between agents, reason over it carefully and provide a final answer.\
                \nHere is the question:{question}\
                \nHere is the summary of solutions made by previous turn of debate from medical experts: \n\n{chat_history_summary[-1]}\
                \nPlease think step-by-step and generate your ourput."
            response = self.moderation_unit.agents['DecisionMaker'].chat(decision_prompt, {
                'Thought': 'Please provide your step-by-step reasoning process for arriving at your answer, '
                            'including: 1) Analysis of the debate summaries and key points of agreement/disagreement between experts, '
                            '2) Evaluation of the evidence and rationale supporting each potential answer, '
                            '3) Explanation of why you believe your chosen answer is correct and why you ruled out other options.',
                'Answer': f'Please pick from {" ".join(["(" + choice + ")" for choice in choices])}.'
            })
            answer_by_turns.append(json.loads(response)['Answer'].replace('(', '').replace(')', ''))
        return answer_by_turns