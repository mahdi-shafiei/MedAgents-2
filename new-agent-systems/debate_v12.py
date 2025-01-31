import os
import re
import json
import time
import utils
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI, OpenAI
from pymilvus import MilvusClient
from datetime import datetime
from tqdm.auto import tqdm
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

retrieval_client = MilvusClient(uri="http://localhost:19530")

#llm_client = OpenAI(
#    api_key=os.getenv("openai_api_key"),
#)

llm_client = AzureOpenAI(
    azure_endpoint="https://azure-openai-miblab-ncu.openai.azure.com/",
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2024-08-01-preview"
)


allowed_sources = ['cpg_2', 'statpearls_2', 'recop_2', 'textbook_2']
sample_size = 1
start_idx = 19
max_workers = 1
llm_debate_max_round = 1
retrieve_topk = 100
rerank_topk = 25
rewrite=False
review=False 
gpu_ids = [0]
voting = 'singular', # 'multi_ranked', 'multi_rated', 'multi_points'

llm_model = "gpt-4o-mini"


class LLM_Agent:
    def __init__(self, domain, llm_client, system_prompt = None):
        self.domain = domain
        if system_prompt is None:
            self.system_prompt = f"You are a medical expert in the domain of {self.domain}."
        else:
            self.system_prompt = system_prompt.format(self.domain)
        self.history = [{'role': 'system', 'content': self.system_prompt}]
        self.llm_client = llm_client

    def chat(self, input_text, return_items = None, save = False):
        if return_items:
            json_format_keys = {key: ("your " + key + ". Please pick from (A), (B), (C) or (D)" if key == 'Answer' else "your " + key) for key in return_items}
            FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\n{json.dumps(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"
            return_instruction = FORMAT_INST(json_format_keys)
            full_input = f"{input_text}\n\n{return_instruction}"
            response = self._generate_response(self.history +[{'role': 'user', 'content': full_input}])
        else: 
            full_input = input_text
        response = self._generate_response(self.history +[{'role': 'user', 'content': full_input}], return_items)
        if save:
                self.history.append({'role': 'user', 'content': full_input})
                self.history.append({'role': 'assistant', 'content': response})
        print("\n--------------FULL_INPUT--------------")
        print(full_input)
        print("\n--------------RESPONSE--------------")
        print(response)
        return response

    def _generate_response(self, messages, return_items=None, max_retries=100) :
        attempt = 0
        while attempt < max_retries:
            try:
                if return_items:
                    response = self.llm_client.chat.completions.create(
                        model = llm_model,
                        messages = messages,
                        max_tokens = 2048,
                        temperature = 0,
                        response_format = { "type": "json_object" }
                    )
                else:
                    response = self.llm_client.chat.completions.create(
                        model = llm_model,
                        messages = messages,
                        max_tokens = 2048,
                        temperature = 0,
                    )
                return response.choices[0].message.content
            except Exception as e:
                if "rate" in str(e).lower() or "exceeded" in str(e).lower():
                    print('rate exceed')
                    attempt += 1
                    wait_time = 30
                    time.sleep(wait_time)
                else:
                    return f"Error generating response: {e}"
        return "Error: Unable to generate response after multiple attempts due to token limit."

    def get_history(self):
        return self.history

class Triage_Unit:
    def __init__(self):
        self.question_classifier_agent = LLM_Agent("Question Triage", llm_client, "You are a medical expert who specializes in categorizing medical scenarios into specific areas of medicine.")
        self.option_classifier_agent = LLM_Agent("Options Triage", llm_client, "As a medical expert, you possess the ability to discern the most relevant fields of expertise needed to address a multiple-choice question encapsulating a specific medical context.")
      
    def _generate_specialties_list_question(self, query: str, medical_fields: List[str], num_fields: int = 5) -> List[str]:
        question_domain_format = "Medical Field: " + " | ".join([f"Field{i}" for i in range(num_fields)])
        prompt_get_question_domain = f"You need to complete the following steps:\n" \
            f"1. Carefully read the medical scenario presented in the question:'''{query}'''.\n" \
            f"2. Based on the medical scenario in it, classify the question into {num_fields} different subfields of medicine: {', '.join(medical_fields)}.\n" \
            f"3. Output exactly in this format: '''{question_domain_format}'''."
        try:
            response = self.question_classifier_agent.chat(prompt_get_question_domain).replace('**', '').replace("\'\'\'", '').strip()
            if "ield: " in response:
                specialty_list = [domain.strip() for domain in response.split("ield: ")[1].split('|') if domain.strip()]
            else:
                specialty_list = [f"General Medicine_{i+1}" for i in range(num_fields)]
        except (IndexError, ValueError, Exception) as e:
            specialty_list = [f"General Medicine_{i+1}" for i in range(num_fields)]
        expert_list = []
        for specialty in specialty_list:
            expert_list.append(LLM_Agent(specialty, llm_client))
        return specialty_list, expert_list

    def _generate_specialties_list_option(self, query: str, options: str, medical_fields: List[str], num_fields: int = 2) -> List[str]:
        option_domain_format = "Medical Field: " + " | ".join([f"Field{i}" for i in range(num_fields)])
        prompt_get_options_domain = f"You need to complete the following steps:" \
            f"1. Carefully read the medical scenario presented in the question: '''{query}'''." \
            f"2. The available options are: '''{options}'''. Strive to understand the fundamental connections between the question and the options." \
            f"3. Your core aim should be to categorize the options into two distinct subfields of medicine. " \
            f"You should output in exactly the same format as '''{option_domain_format}'''"
        try:
            response = self.option_classifier_agent.chat(prompt_get_question_domain).replace('**', '').replace("\'\'\'", '').strip()
            if "ield: " in response:
                specialty_list = [domain.strip() for domain in response.split("ield: ")[1].split('|') if domain.strip()]
        except (IndexError, ValueError, Exception) as e:
            specialty_list = [f"General Medicine_{chr(65 + i)}" for i in range(num_fields)]
        expert_list = []
        for specialty in specialty_list:
            expert_list.append(LLM_Agent(specialty))
        return specialty_list, expert_list

class Search_Unit:
    def __init__(self, retrieval_client, retrieve_topk, rerank_topk, allowed_sources):
        self.query_formulate_agent = LLM_Agent("Query Rewriter", llm_client, "Write a medical passage that can help answer the given query. Include key information or terminology for the answer.")
        self.document_evaluate_agent = LLM_Agent("Document Evaluator", llm_client, "You are an expert in evaluating the relevance of documents to a given query.")
        self.retrieve_topk = retrieve_topk
        self.rerank_topk = rerank_topk
        self.allowed_sources = allowed_sources
        self.oom_count = 0

    def _retrieve_query(self, query) -> List[str]:
        retrieved_docs = []
        try:
            docs = retrieve(query, retrieval_client, self.retrieve_topk, self.rerank_topk)
            retrieved_docs.extend(docs[:self.rerank_topk])
        except Exception as e:
            if "memory" in str(e).lower():
                print('retrieve memory error')
                self.oom_count += 1
        return retrieved_docs

    def _review_documents(self, query, documents) -> List[str]:
        reviewed_docs = []
        review_prompt = "Evaluate the relevance of the following document to the given query" \
        "Instructions:"\
        "1. Assess the document's helpfulness in answering the query."\
        "2. Label the document as one of the following:"\
        "    - [Fully Helpful]: Contains comprehensive and relevant information directly addressing the query."\
        "    - [Partially Helpful]: Contains some relevant information but lacks full coverage or precision."\
        "    - [Not Helpful]: Contains no useful information for the query."\
        "3. Only respond with one of the labels: [Fully Helpful], [Partially Helpful], [Not Helpful]."\
        "Query: {}"\
        "Document: {}"
        for doc in documents:
            try:
                response = document_evaluate_agent.chat(review_prompt.format(query, doc))
                if any(substring in response.lower() for substring in ["ully", "artially"]):
                    reviewed_docs.append(doc)
            except Exception as e:
                continue
        return reviewed_docs

    def _rewrite_query_pseudodoc(self, query):
        few_shot_prompt = f"Write a medical passage that can help answer the given query. Include key information or terminology for the answer."\
            f""\
            f"Example:"\
            f""\
            f"Query: A 39-year-old woman presents to the family medicine clinic to be evaluated by her physician for weight gain. She reports feeling fatigued most of the day despite eating a healthy diet and exercising regularly. The patient smokes a half-pack of cigarettes daily and has done so for the last 23 years. She is employed as a phlebotomist by the Red Cross. She has a history of hyperlipidemia for which she takes atorvastatin. She is unaware of her vaccination history, and there is no documented record of her receiving any vaccinations. Her heart rate is 76/min, respiratory rate is 14/min, temperature is 37.3°C (99.1°F), body mass index (BMI) is 33 kg/m2, and blood pressure is 128/78 mm Hg. The patient appears alert and oriented. Lung and heart auscultation are without audible abnormalities. The physician orders a thyroid panel to determine if that patient has hypothyroidism. Which of the following recommendations may be appropriate for the patient at this time? A) Hepatitis B vaccination B) Low-dose chest CT C) Hepatitis C vaccination D) Shingles vaccination"\
            f"Passage: against vaccine-preventable diseases. Every visit by an adult to a health-care provider should be an opportunity to provide this protection. Several factors need to be con sidered before any patient is vaccinated. These include the susceptibility of the patient, the risk of exposure to the disease, the risk from the disease, and the benefits and risks from the immunizing agent. Physicians should maintain detailed information about previous vaccina tions received by each individual, including type of vaccination, date of receipt, and adverse events, if any, following vaccination. Information should also include the person's history of vaccine-preventable illnesses, occupation, and lifestyle. Vaccine histories ideally should be based on written documentation to ascertain whether vaccines and toxoids were administered at appropriate ages and at proper intervals. Close attention to factors"\
            f""\
            f"Query: A 23-year-old male presents to his primary care physician after an injury during a rugby game. The patient states that he was tackled and ever since then has had pain in his knee. The patient has tried NSAIDs and ice to no avail. The patient has no past medical history and is currently taking a multivitamin, fish oil, and a whey protein supplement. On physical exam you note a knee that is heavily bruised. It is painful for the patient to bear weight on the knee, and passive motion of the knee elicits some pain. There is laxity at the knee to varus stress. The patient is wondering when he can return to athletics. Which of the following is the most likely diagnosis? A) Medial collateral ligament tear B) Lateral collateral ligament tear C) Anterior cruciate ligament tear D) Posterior cruciate ligament tear"\
            f"Passage: Diagnosing PCL Injuries: History, Physical Examination, Imaging Studies, Arthroscopic Evaluation. Isolated posterior cruciate ligament (PCL) injuries are uncommon and can be easily missed with physical examination. The purpose of this article is to give an overview of the clinical, diagnostic and arthroscopic evaluation of a PCL injured knee. There are some specific injury mechanisms that can cause a PCL including the dashboard direct anterior blow and hyperflexion mechanisms. During the diagnostic process it is important to distinguish between an isolated or multiligament injury and whether the problem is acute or chronic. Physical examination can be difficult in an acutely injured knee because of pain and swelling, but there are specific functional tests that can indicate a PCL tear. Standard x-ray's and stress views are very useful imaging modalities"\
            f""\
            f"Query: A 45-year-old woman is in a high-speed motor vehicle accident and suffers multiple injuries to her extremities and abdomen. In the field, she was bleeding profusely bleeding and, upon arrival to the emergency department, she is lethargic and unable to speak. Her blood pressure on presentation is 70/40 mmHg. The trauma surgery team recommends emergency exploratory laparotomy. While the patient is in the trauma bay, her husband calls and says that the patient is a Jehovah's witness and that her religion does not permit her to receive a blood transfusion. No advanced directives are available. Which of the following is an appropriate next step? A) Provide transfusions as needed B) Withhold transfusion based on husband's request C) Obtain an ethics consult D) Obtain a court order for transfusion"\
            f"Passage: Legal and ethical issues in safe blood transfusion. This is another D and C Act requirement which is seldom followed, possibly because there are no standard guidelines."\
            f""\
            f"Query: A 4-year-old male is accompanied by his mother to the pediatrician. His mother reports that over the past two weeks, the child has had intermittent low grade fevers and has been more lethargic than usual. The child’s past medical history is notable for myelomeningocele complicated by lower extremity weakness as well as bowel and bladder dysfunction. He has been hospitalized multiple times at an outside facility for recurrent urinary tract infections. The child is in the 15th percentile for both height and weight. His temperature is 100.7°F (38.2°C), blood pressure is 115/70 mmHg, pulse is 115/min, and respirations are 20/min. Physical examination is notable for costovertebral angle tenderness that is worse on the right. Which of the following would most likely be found on biopsy of this patient’s kidney? A) Mononuclear and eosinophilic infiltrate B) Replacement of renal parenchyma with foamy histiocytes C) Destruction of the proximal tubule and medullary thick ascending limb D) Tubular colloid casts with diffuse lymphoplasmacytic infiltrate"\
            f"Passage: The natural history of urinary infection in adults. The vast majority of otherwise healthy adults with anatomically and functionally normal urinary tracts experience few untoward long-term consequences from symptomatic or asymptomatic UTIs. Effective early treatment of symptomatic infection rapidly curtails bacterial invasion and the resulting inflammatory response. Rarely, uncomplicated acute pyelonephritis causes suppuration and renal scarring. Urinary infections in patients with renal calculi, obstructed urinary tract, neurogenic bladder, or diabetes are frequently much more destructive and have ongoing sequelae. Strategies to treat both the infection and the complications are often necessary to alter this outcome."\
            f"..."\
            f""\
            f"Query: {query}"\
            f"Passage:"
        return self.query_formulate_agent.chat( )

    def get_oom_count(self):
        return self.oom_count

class Moderation_Unit:
    def __init__(self):
        self.moderator = LLM_Agent("Moderator", llm_client, system_prompt = "You are a skilled and impartial moderator overseeing a medical debate.")
        self.decider = LLM_Agent("Final_Decider", llm_client, system_prompt = "You are a knowledgeable medical expert.")         

class Discussion_Unit:
    def __init__(self, question, expert_list, search_unit, moderation_unit, max_iterations=5):
        self.question = question
        self.agents = {idx: agent for idx, agent in enumerate(expert_list)}
        self.search_unit = search_unit
        self.moderation_unit = moderation_unit
        self.max_iterations = max_iterations
        self.q_a_pairs = []
#        self.answers_table = self._initialize_answers_table()
 
    def _decompose_query(self, question, agent, qa_pairs):
        decompose_prompt = f"Main question to solve: {question}\n"
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
    
        response = agent.chat(decompose_prompt, ['Query'])
        decomposed_query = json.loads(response)['Query']
        return decomposed_query

    def _request_search(self, question, rewrite=False, review=False):
        og_question = question
        if rewrite:
            question = self.search_unit._rewrite_query_pseudodoc(question)
        documents = self.search_unit._retrieve_query(question)
        if review:
            documents = self.search_unit._review_documents(og_question, documents)
        return documents
    
    def _decomposed_rag(self, question, rewrite=False, review=False):
        for idx, agent in self.agents.items():
            decomposed_query = self._decompose_query(question, agent, self.q_a_pairs)
            if rewrite == "Both":
                decomposed_documents = (
                    self._request_search(decomposed_query, rewrite=False, review=review)
                    + self._request_search(decomposed_query, rewrite=True, review=review)
                )
            else:
                decomposed_documents = self._request_search(decomposed_query, rewrite=rewrite, review=review)
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

    def simultaneous_talk_summarizer_v2(self, max_round):
        answer_by_turns = []
        chat_history = [[] for _ in range(max_round)]
        chat_history_summary = []
        og_documents = '\n'.join(self._request_search(self.question, rewrite=True, review=review))
        for r in range(max_round):
            print("-"*50)
            print(f"{r}th round debate start")
            print("-"*50)
            for i, agent in self.agents.items():
                if r == 0:
                    user_prompt = f"The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer.\n\n"
                    user_prompt += f"Here is the relevant document: {og_documents}\n\n"
                    user_prompt += f"Here is the question: {self.question}\n\n"
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
                    user_prompt += f"Here is the question:{self.question}\n"
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
            summary = self.moderation_unit.moderator.chat(summarizer_prompt)
            chat_history_summary.append(summary)

            summaries = '\n'.join([f"{i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd' if i == 2 else 'th'} debate summary: {chat}" for i, chat in enumerate(chat_history_summary)])
            decider_prompt = f"Given the summary of previous debate between agents, reason over it carefully and provide a final answer.\
                \nHere is the question:{self.question}\
                \nHere is the summary of solutions made by previous turn of debate from medical experts: \n\n{chat_history_summary[-1]}\
                \nPlease think step-by-step and generate your ourput."
            response = self.moderation_unit.decider.chat(decider_prompt, ['Thought', 'Answer'])
            answer_by_turns.append(json.loads(response)['Answer'].replace('(', '').replace(')', ''))
        return answer_by_turns

medqa_test = []
with open(f"{parent_dir}/data/medqa/test_hard.jsonl", 'r') as jsfile:
    for line in jsfile:
        medqa_test.append(json.loads(line))
medqa_test = medqa_test[start_idx:start_idx+sample_size]
queries = [f"{test['question']}\n\nOptions: (A) {test['options']['A']} (B) {test['options']['B']} (C) {test['options']['C']} (D) {test['options']['D']}" for test in medqa_test]
results = [None] * len(queries)

def process_query(query, task_number):
    triage_unit = Triage_Unit()
    specialty_list, expert_list = triage_unit._generate_specialties_list_question(query, utils.medical_specialties_gpt_selected, 5)
    search_unit = Search_Unit(retrieval_client, retrieve_topk, rerank_topk, allowed_sources)
    moderation_unit = Moderation_Unit()
    discussion_unit = Discussion_Unit(query, expert_list, search_unit, moderation_unit, llm_debate_max_round)
    discussion_unit._decomposed_rag(query, rewrite, review)
    results = discussion_unit.simultaneous_talk_summarizer_v2(llm_debate_max_round)
    return results, search_unit.get_oom_count()

if __name__ == "__main__":

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Include script name in the output directory
    output_dir = (
        f"./output/{script_name}/"
        f"results_llm_{llm_model}_rounds_{llm_debate_max_round}_retrieve_{retrieve_topk}_"
        f"rerank_{rerank_topk}_rewrite_{rewrite}_review_{review}_time_{current_time}"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    oom_count_total = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(process_query, query, task_number): task_number
            for task_number, query in enumerate(queries)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing queries"):
            idx = future_to_index[future]
            try:
                result, oom_count = future.result()  
                results[idx] = result
                oom_count_total += oom_count
            except Exception as e:
                print(f"Error: {str(e)}")
                results[idx] = f"Error: {str(e)}"  

    print(f"Total OOM errors encountered: {oom_count_total}")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    with open(os.path.join(output_dir, "results.json"),'w') as jsfile:
        json.dump(results, jsfile)
    for debate_round in range(llm_debate_max_round):
        count = 0
        for i in range(len(medqa_test)):
            if results[i][debate_round] == medqa_test[i]['answer_idx']:
                count += 1
        print(count/len(medqa_test))








