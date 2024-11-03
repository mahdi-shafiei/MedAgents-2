import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from pymilvus import MilvusClient
from dotenv import load_dotenv
import utils
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

load_dotenv()

retrieval_client = MilvusClient(uri="http://localhost:19530")
llm_client = AzureOpenAI(
    azure_endpoint="https://azure-openai-miblab-ncu.openai.azure.com/",
    api_key=os.getenv("azure_api_key"),
    api_version="2024-08-01-preview"
)

sample_size = 50
start_idx = 0

class CIDER:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.stacked_retrieved_document = []
        self.iteration_history = []
        self.expert_roles = []
        self.retrieve_topk = 100
        self.rerank_topk = 25
        self.max_iterations = 3
        self.doc_length = 50000
        self.query_temperature = 0

    def process_query(self, initial_query: str) -> Dict[str, Any]:
        iteration = 0
        while iteration < self.max_iterations:
            if iteration == 0:
                # 1. Role
                self.expert_roles = self._generate_expert_domains(initial_query, utils.medical_specialties_gpt_selected)
               # 2. Query
                queries = self._generate_expert_query(['general medicine'], initial_query)
            else:
                queries = self._generate_disagreement_resolution_query(initial_query, self.iteration_history)
            # 3. Retrieve
            retrieved_docs = self._retrieve_queries(queries)
            if iteration ==0:
                retrieved_docs = self._filter_docs(initial_query, retrieved_docs, model = "gpt-4o-mini")
            else:
                retrieved_docs = self._filter_docs(queries[0], retrieved_docs, model = "gpt-4o-mini")
            if retrieved_docs:
                self._update_documents(retrieved_docs)
            # 4. Analize
            if iteration == 0:
                expert_analyses = self._expert_analysis(self.expert_roles, retrieved_docs, initial_query)
                #expert_analyses.append(self._cot_generalmedicine(initial_query))
            else:
                expert_analyses = self._expert_resolution(self.expert_roles, retrieved_docs, initial_query, self.iteration_history)
            # 5. Synthesize
            if iteration == 0:
                synthesized_result = self._synthesize_results(initial_query, expert_analyses)
            else: 
                synthesized_result = self._synthesize_resolution(initial_query, expert_analyses, self.iteration_history)
            # 6. Consensus
            consensus_result = self._check_consensus(initial_query, expert_analyses, self.iteration_history)
            # 7. Record
            self.iteration_history.append({
                'iteration': iteration,
                'queries': queries,
                'docs': retrieved_docs,
                'analyses': expert_analyses,
                'reports': synthesized_result,
                'consensus': consensus_result
            })
            if "ensus: yes" in consensus_result.lower():
                return self._final_answer_pick(initial_query,synthesized_result + "\n" + consensus_result)
            else:
                iteration += 1
        return self._final_answer_pick(initial_query,synthesized_result + "\n" + consensus_result)


    def _clean_text(self, text: str) -> str:
        """Remove markdown formatting and clean text."""
        if not isinstance(text, str):
            return ""
        return text.replace('**', '').replace("\'\'\'", '').strip()

    def _call_llm(self, messages, model, temperature = 0,) -> str:
        response = llm_client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=10000
        )
        print("\n" + "="*50 + " LLM Input Messages " + "="*50)
        for message in messages:
            print(f"{message['role']}: {message['content']}\n")
        output = response.choices[0].message.content
        print("\n" + "="*50 + " LLM Output Response " + "="*50)
        print(output + "\n")
        return self._clean_text(output)

    def _update_documents(self, new_documents: List[str]):    
        added_documents = []
        for document in new_documents:
            if document not in self.stacked_retrieved_document:
                self.stacked_retrieved_document.append(document)
                added_documents.append(document)
#        combined_docs = "\n".join(set(new_documents) - set(self.current_knowledge))
#        self.stacked_retrieved_document.append(combined_docs)
        return added_documents


    def _summarize_documents(self, documents: str, original_query: str) -> str:
        summary_prompt = f"""Create a document that contains key information from the following set of medical documents, considering their relevance to the original query:

        Original Query: {original_query}

        Documents: {documents}

        Please provide a medical document with clinical information that might be helpful for answering the query. Do not suggest or conclude any answer choice."""
        try:
            return self._clean_text(self._call_llm([
                {"role": "system", "content": "You are a medical document writer including key information from documents."},
                {"role": "user", "content": summary_prompt}
            ],
            self.model,
            temperature = 0.0
            ))
        except Exception as e:
            print(f"Error summarizing documents: {str(e)}")
            return ""

    def assess_query_difficulty(self, query: str) -> str:
        """Assess the complexity of a medical query."""
        difficulty_prompt = """Given the medical query below, assess its difficulty:

        Query: {query}

        Options:
        - Easy: A single medical agent can answer it.
        - Hard: A multi-agent system is needed.

        Format your response:
        Explanation: <Brief explanation>
        Difficulty: <easy/hard>"""

        response = self._call_llm([
            {"role": "system", "content": "You are a medical expert evaluating query complexity."},
            {"role": "user", "content": difficulty_prompt.format(query=query)}
        ],
        self.model,
        temperature = 0)
        return 'hard' if 'hard' in response.lower() else 'easy'

    def _generate_expert_domains(self, query: str, medical_fields: List[str], num_fields: int = 5) -> List[str]:
        """Generate relevant expert domains for the query."""
        question_domain_format = "Medical Field: " + " | ".join([f"Field{i}" for i in range(num_fields)])
        question_classifier = "You are a medical expert who specializes in categorizing medical scenarios into specific areas of medicine."
        prompt_get_question_domain = (
            f"Complete these steps:\n"
            f"1. Read the medical scenario: '''{query}'''.\n"
            f"2. Classify into these subfields: {', '.join(medical_fields)}.\n"
            f"3. Output exactly in this format: '''{question_domain_format}'''."
        )

        try:
            response = self._clean_text(self._call_llm([
                {"role": "system", "content": question_classifier},
                {"role": "user", "content": prompt_get_question_domain}
            ], self.model,temperature = 0
            ))

            if "ield: " in response:
                domain_list = [domain.strip() for domain in response.split("ield: ")[1].split('|') if domain.strip()]
                if len(domain_list) == 1:
                    domain_list.append('General Medicine')
                return domain_list
            raise ValueError("Delimiter 'ield: ' not found in the response.")
        except (IndexError, ValueError, Exception) as e:
            return ['General Medicine'] * num_fields

    def _generate_expert_query(self, roles, context) -> List[str]:
        query_prompt = """Example:

Query: A 39-year-old woman presents to the family medicine clinic to be evaluated by her physician for weight gain. She reports feeling fatigued most of the day despite eating a healthy diet and exercising regularly. The patient smokes a half-pack of cigarettes daily and has done so for the last 23 years. She is employed as a phlebotomist by the Red Cross. She has a history of hyperlipidemia for which she takes atorvastatin. She is unaware of her vaccination history, and there is no documented record of her receiving any vaccinations. Her heart rate is 76/min, respiratory rate is 14/min, temperature is 37.3°C (99.1°F), body mass index (BMI) is 33 kg/m2, and blood pressure is 128/78 mm Hg. The patient appears alert and oriented. Lung and heart auscultation are without audible abnormalities. The physician orders a thyroid panel to determine if that patient has hypothyroidism. Which of the following recommendations may be appropriate for the patient at this time? A) Hepatitis B vaccination B) Low-dose chest CT C) Hepatitis C vaccination D) Shingles vaccination
Passage: against vaccine-preventable diseases. Every visit by an adult to a health-care provider should be an opportunity to provide this protection. Several factors need to be con sidered before any patient is vaccinated. These include the susceptibility of the patient, the risk of exposure to the disease, the risk from the disease, and the benefits and risks from the immunizing agent. Physicians should maintain detailed information about previous vaccina tions received by each individual, including type of vaccination, date of receipt, and adverse events, if any, following vaccination. Information should also include the person's history of vaccine-preventable illnesses, occupation, and lifestyle. Vaccine histories ideally should be based on written documentation to ascertain whether vaccines and toxoids were administered at appropriate ages and at proper intervals. Close attention to factors

Query: A 23-year-old male presents to his primary care physician after an injury during a rugby game. The patient states that he was tackled and ever since then has had pain in his knee. The patient has tried NSAIDs and ice to no avail. The patient has no past medical history and is currently taking a multivitamin, fish oil, and a whey protein supplement. On physical exam you note a knee that is heavily bruised. It is painful for the patient to bear weight on the knee, and passive motion of the knee elicits some pain. There is laxity at the knee to varus stress. The patient is wondering when he can return to athletics. Which of the following is the most likely diagnosis? A) Medial collateral ligament tear B) Lateral collateral ligament tear C) Anterior cruciate ligament tear D) Posterior cruciate ligament tear
Passage: Diagnosing PCL Injuries: History, Physical Examination, Imaging Studies, Arthroscopic Evaluation. Isolated posterior cruciate ligament (PCL) injuries are uncommon and can be easily missed with physical examination. The purpose of this article is to give an overview of the clinical, diagnostic and arthroscopic evaluation of a PCL injured knee. There are some specific injury mechanisms that can cause a PCL including the dashboard direct anterior blow and hyperflexion mechanisms. During the diagnostic process it is important to distinguish between an isolated or multiligament injury and whether the problem is acute or chronic. Physical examination can be difficult in an acutely injured knee because of pain and swelling, but there are specific functional tests that can indicate a PCL tear. Standard x-ray's and stress views are very useful imaging modalities

Query: A 45-year-old woman is in a high-speed motor vehicle accident and suffers multiple injuries to her extremities and abdomen. In the field, she was bleeding profusely bleeding and, upon arrival to the emergency department, she is lethargic and unable to speak. Her blood pressure on presentation is 70/40 mmHg. The trauma surgery team recommends emergency exploratory laparotomy. While the patient is in the trauma bay, her husband calls and says that the patient is a Jehovah's witness and that her religion does not permit her to receive a blood transfusion. No advanced directives are available. Which of the following is an appropriate next step? A) Provide transfusions as needed B) Withhold transfusion based on husband's request C) Obtain an ethics consult D) Obtain a court order for transfusion
Passage: Legal and ethical issues in safe blood transfusion. This is another D and C Act requirement which is seldom followed, possibly because there are no standard guidelines.

Query: A 4-year-old male is accompanied by his mother to the pediatrician. His mother reports that over the past two weeks, the child has had intermittent low grade fevers and has been more lethargic than usual. The child’s past medical history is notable for myelomeningocele complicated by lower extremity weakness as well as bowel and bladder dysfunction. He has been hospitalized multiple times at an outside facility for recurrent urinary tract infections. The child is in the 15th percentile for both height and weight. His temperature is 100.7°F (38.2°C), blood pressure is 115/70 mmHg, pulse is 115/min, and respirations are 20/min. Physical examination is notable for costovertebral angle tenderness that is worse on the right. Which of the following would most likely be found on biopsy of this patient’s kidney? A) Mononuclear and eosinophilic infiltrate B) Replacement of renal parenchyma with foamy histiocytes C) Destruction of the proximal tubule and medullary thick ascending limb D) Tubular colloid casts with diffuse lymphoplasmacytic infiltrate
Passage: The natural history of urinary infection in adults. The vast majority of otherwise healthy adults with anatomically and functionally normal urinary tracts experience few untoward long-term consequences from symptomatic or asymptomatic UTIs. Effective early treatment of symptomatic infection rapidly curtails bacterial invasion and the resulting inflammatory response. Rarely, uncomplicated acute pyelonephritis causes suppuration and renal scarring. Urinary infections in patients with renal calculi, obstructed urinary tract, neurogenic bladder, or diabetes are frequently much more destructive and have ongoing sequelae. Strategies to treat both the infection and the complications are often necessary to alter this outcome.
...

Query: {}
Passage:"""
        all_queries = []
        for role in roles:
            try:
                response = self._clean_text(self._call_llm([
                    {"role": "system", "content": f"You are a medical expert in {role}. Write a medical passage that can help answer the given query. Include key information or terminology for the answer."},
                    {"role": "user", "content": query_prompt.format(context)}
                    ],self.model,temperature=0)
                    )
                all_queries.append(response)
            except Exception as e:
                print(f"Error generating queries for {role}: {str(e)}")
        return list(set(all_queries[:]))

    def _expert_analysis(self, roles, documents_or_knowledgebase: List[str], query: str) -> List[Dict]:
        output_report_format = "Key Knowledge: [extracted key knowledge]\nTotal Analysis: [synthesized analysis]"

        prompt = (
            f"Please meticulously examine the medical scenario outlined in this question: '''{query}'''.\n"
            f"Drawing upon your medical expertise, interpret the condition being depicted.\n"
            f"Subsequently, identify and highlight the aspects of the issue that you find most alarming or noteworthy.\n\n"
            f"Output Format:\n{output_report_format}\n"
        )

        formatted_documents = self._format_docs_for_prompt(documents_or_knowledgebase)
        truncated_docs = formatted_documents[:self.doc_length] 
        analysis_prompt = (
            f"{prompt}\n"
            f"Relevant Documents: {truncated_docs}\n\n"
        )

        return [
            {
                'role': role,
                'analysis': self._clean_text(self._call_llm([
                    {"role": "system", "content": f"You are a {role} specialist trying to solve a medical question by analyzing the context and documents."},
                    {"role": "user", "content": analysis_prompt}
                ],self.model,temperature=0))
            }
            for role in roles
        ]

    def _expert_resolution(self, roles, documents_or_knowledgebase, query, history) -> List[Dict]:
        formatted_documents = self._format_docs_for_prompt(documents_or_knowledgebase)
        truncated_docs = formatted_documents[:self.doc_length]
        history_entry = history[-1]
        analyses_text = '\n\n'.join([f"Expert ({a['role']}):\n{a['analysis']}" for a in history_entry['analyses']])
        analysis_prompt = (
            f"The medical experts previously disagreed on the following medical question: {query}\n\n"
            f"Previous Expert Analyses:\n{analyses_text}\n\n"
            f"Previous Report Made from Experts: {history_entry['reports']}\n\n"
            f"Consensus Report: {history_entry['consensus']}\n\n"
            f"Here is the new information gathered to resolve the disagreements: {formatted_documents}\n\n"
            f"Reflect on your previous thoughts and analyses. Please analyze whether the new information addresses the disagreements and helps reach a consensus.\n"
            f"If the disagreements are resolved, explain how the new information has helped. If disagreements remain, describe what further information might be needed."
        )
        return [
            {
                'role': role,
                'analysis': self._clean_text(self._call_llm([
                    {"role": "system", "content": f"You are a {role} specialist trying to solve a medical question by analyzing the context and documents."},
                    {"role": "user", "content": analysis_prompt}
                ],self.model,temperature=0))
            }
            for role in roles
        ]
        return response




    def _cot_generalmedicine(self, query: str) -> Dict:
        return {
            'role': "General Medicine",
            'analysis': self._clean_text(self._call_llm([
                {"role": "system", "content": "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\""},
                {"role": "user", "content": f"Question: {query}"}
            ],self.model,temperature=0))
        }

    def _retrieve_queries(self, queries: List[str]) -> List[str]:
        retrieved_docs = []
        for query in queries:
            docs = utils.rerank(query, utils.retrieve_filtered_sources(query, retrieval_client, topk = self.retrieve_topk))
            retrieved_docs.extend(docs[:self.rerank_topk])
        seen = set()
        seen_add = seen.add
        return [x for x in retrieved_docs if not (x in seen or seen_add(x))]

    def _filter_docs(self, query: str, retrieved_docs: List[str], model) -> List[str]:
        filtered_docs = []

        filter_prompt = """Evaluate the relevance of the following document to the given query.

        Query: {query}

        Document: {doc}

        Instructions:
        1. Assess the document's helpfulness in answering the query.
        2. Label the document as one of the following:
            - [Fully Helpful]: Contains comprehensive and relevant information directly addressing the query.
            - [Partially Helpful]: Contains some relevant information but lacks full coverage or precision.
            - [Not Helpful]: Contains no useful information for the query.
        3. Only respond with one of the labels: [Fully Helpful], [Partially Helpful], [Not Helpful].
        """

        for doc in retrieved_docs:
            if doc in self.stacked_retrieved_document:
                continue
            try:
                response = self._clean_text(self._call_llm([
                    {"role": "system", "content": "You are an expert in evaluating the relevance of documents to a given query."},
                    {"role": "user", "content": filter_prompt.format(query=query, doc=doc)}
                ], model,temperature=0))
                if any(substring in response.lower() for substring in ["ully", "artially"]):
                    filtered_docs.append(doc)
            except Exception as e:
                print(f"Error filtering document: {str(e)}")
                continue
        return filtered_docs

    def _synthesize_results(self, query, expert_analyses: List[Dict]) -> str:
        synthesizer = "You are a medical decision maker who excels at summarizing and synthesizing based on multiple experts from various domain experts."

        syn_report_format = "Key Knowledge: [extracted key knowledge] \nTotal Analysis: [synthesized analysis] \n"
        expert_analyses_text = '\n\n'.join([f"Expert ({a['role']}):\n{a['analysis']}" for a in expert_analyses])
        prompt = (
            f"Here are some reports from different medical domain experts.\n"
            f"You need to complete the following steps:\n"
            f"1. Take careful and comprehensive consideration of the following reports.\n"
            f"2. Extract key knowledge from the following reports.\n"
            f"3. Derive the comprehensive and summarized analysis based on the knowledge.\n"
            f"4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n"
            f"You should output in exactly the same format as '''{syn_report_format}'''\n\n"
            f"Expert Analyses:\n{expert_analyses_text}\n\n"
            f"Original Query:\n{query}\n"
        )

        response = self._clean_text(self._call_llm([
            {"role": "system", "content": synthesizer},
            {"role": "user", "content": prompt}
        ],self.model,temperature=0))

        return response

    def _synthesize_resolution(self, query, expert_analyses: List[Dict], history) -> str:
        history_entry = history[-1]
        synthesizer = "You are a medical decision maker who excels at summarizing and synthesizing based on multiple experts from various domain experts."
        syn_report_format = "Key Knowledge: [extracted key knowledge] \nTotal Analysis: [synthesized analysis] \n"
        expert_analyses_text = '\n\n'.join([f"Expert ({a['role']}):\n{a['analysis']}" for a in expert_analyses])

        prompt = (
            f"Here are some reports from different medical domain experts.\n"
            f"You need to complete the following steps:\n"
            f"1. Take careful and comprehensive consideration of the following reports.\n"
            f"2. Extract key knowledge from the following reports.\n"
            f"3. Compare this new analysis with the previous report and consensus report.\n"
            f"4. Derive a comprehensive and synthesized analysis based on the new insights and reflect on how the new information has changed the consensus, if at all.\n"
            f"5. Your ultimate goal is to derive a refined and synthesized report based on the following reports, taking into account previous consensus and changes brought by new analyses.\n"
            f"You should output in exactly the same format as '''{syn_report_format}'''\n\n"
            f"Expert Analyses:\n{expert_analyses_text}\n\n"
            f"Previous Report:\n{history_entry['reports']}\n\n"
            f"Previous Consensus Report:\n{history_entry['consensus']}\n\n"
            f"Original Query to Solve:\n{query}\n"
        )
        response = self._clean_text(self._call_llm([
            {"role": "system", "content": synthesizer},
            {"role": "user", "content": prompt}
        ],self.model,temperature=0))
        return response



    def _check_consensus(self, query, expert_analyses: List[Dict], history) -> str:
        consensus_prompt = """Review the expert opinions below and determine whether they reach a consensus.

        Instructions:
        1. Indicate if there is a consensus: <yes/no>  (Consensus is achieved if ALL experts agree on the same answer choice, regardless of whether the answer is correct.)
        2. If there is a consensus, provide the agreed-upon answer choice: <state the answer>
        3. If there is no consensus, describe the disagreements and suggest areas to search for more information to clarify the correct answer.

        Expert Analyses:
        {a}

        Original Query:
        {q}
        
        Please follow the output format:

        1. Consensus: yes or no
        2. if answer, 
            The answer is: (A), (B), (C), or (D) answer
            if no consensus,
            Disagreements: detailed disagreements
        """

        response = self._clean_text(self._call_llm([
            {"role": "system", "content": "You are a moderator evaluating expert opinions for consensus on a question."},
            {"role": "user", "content": consensus_prompt.format(
                a='\n\n'.join([f"Expert ({a['role']}):\n{a['analysis']}" for a in expert_analyses]),
                q=query
            )}
        ],self.model,temperature=0))

        return response

    def _generate_disagreement_resolution_query(self, query, history) -> str:
        history_entry = history[-1]
        consensus_prompt = (
            f"Medical Experts were presented with this question: {query}\n\n"
            f"They collaboratively generated a report with the following findings and arguments: {history_entry['reports']}\n\n"
            f"After discussing the findings, the experts reached a consensus analysis as follows: {history_entry['consensus']}\n\n"
            f"There are still disagreements, uncertainties, or differing viewpoints in their analysis. Please generate a specific and targeted query, including key terminologies relevant to the topic, to retrieve additional information or evidence that could address the disagreements, clarify uncertainties, or help achieve a more definitive consensus on the matter. Be precise and take into account the current disagreements presented.\n\n"
            f"Query:"
        )

        response = self._clean_text(self._call_llm([
            {"role": "system", "content": "You are an expert query writer, specializing in generating precise queries to gather evidence that could resolve conflicting expert opinions. Make sure to include key terminologies to enhance the relevance and effectiveness of the query."},
            {"role": "user", "content": consensus_prompt}
        ], self.model, temperature=0))

        return [response]


        return [response]

    def _identify_knowledge_gaps(self) -> str:
        if not self.iteration_history:
            return "Initial query - no gaps identified"

        last_iteration = self.iteration_history[-1]
        gaps = []
        for analysis in last_iteration['analyses']:
            try:
                if analysis.get('analysis') and 'Remaining Questions' in analysis['analysis']:
                    questions_part = analysis['analysis'].split('Remaining Questions:')
                    if len(questions_part) > 1:
                        questions = [
                            self._clean_text(q)
                            for q in questions_part[1].split('\n') 
                            if self._clean_text(q)
                        ]
                        gaps.extend(questions)
            except Exception as e:
                print(f"Error processing analysis for gaps: {str(e)}")
                continue
    
        return '\n'.join(gaps) if gaps else "No specific gaps identified"

    def _format_docs_for_prompt(self, documents: List[str]) -> str:
        """Format documents for inclusion in prompts."""
        return "\n\n".join([f"Document {i+1}:\n{self._clean_text(doc)}" for i, doc in enumerate(documents)])

    def _final_answer_pick(self, query, text):
        return self._call_llm([
                    {"role": "system", "content": f"You are an answer parser. Pick an answer even if there is no consensus."},
                    {"role": "user", "content": f"Only output A, B, C, or D from the context: {text}\n\n initial question and options: {query}"}
                ],self.model,temperature=0)

medqa_test = []
with open("datasets/MedQA/50_sampled_hard_medqa/test.jsonl", 'r') as jsfile:
    for line in jsfile:
        medqa_test.append(json.loads(line))

medqa_test = medqa_test[start_idx:start_idx+sample_size]

queries = [f"{test['question']}\n\nOptions: (A) {test['options']['A']} (B) {test['options']['B']} (C) {test['options']['C']} (D) {test['options']['D']}" for test in medqa_test]
cider = CIDER()

results = [None] * len(queries)
with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_index = {executor.submit(cider.process_query, query): idx for idx, query in enumerate(queries)}
    for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing queries"):
        idx = future_to_index[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            results[idx] = f"Error: {str(e)}"

script_name = os.path.splitext(os.path.basename(__file__))[0]
with open(f"/data/jiwoong/workspace/output/result_{script_name}",'w') as jsfile:
    json.dump(results, jsfile)

count = 0
for i in range(len(medqa_test)):
    if results[i][0] == medqa_test[i]['answer_idx']:
        count += 1

print(count/len(medqa_test))
