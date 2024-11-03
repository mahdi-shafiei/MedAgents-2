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

sample_size = 50#50
start_idx = 0

class CIDER:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.current_knowledge = []
        self.iteration_history = []
        self.expert_roles = []
        self.retrieve_topk = 100
        self.rerank_topk = 100
        self.max_iterations = 3
        self.doc_length = 50000

    def _clean_text(self, text: str) -> str:
        """Remove markdown formatting and clean text."""
        if not isinstance(text, str):
            return ""
        return text.replace('**', '').replace("\'\'\'", '').strip()

    def _call_llm(self, messages: List[Dict]) -> str:
        """Make an API call to the LLM and return the response."""
        response = llm_client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=10000
        )
        print("\n" + "="*50 + " LLM Input Messages " + "="*50)
        for message in messages:
            print(f"{message['role']}: {message['content']}\n")
        output = response.choices[0].message.content
        print("\n" + "="*50 + " LLM Output Response " + "="*50)
        print(output + "\n")
        return self._clean_text(output)

    def process_query(self, initial_query: str) -> Dict[str, Any]:

        iteration = 0
        max_iterations = self.max_iterations

        while iteration < max_iterations:
            if iteration == 0:
                self.expert_roles = self._generate_expert_domains(initial_query, utils.medical_specialties_gpt_selected)
                queries = self._generate_expert_query(initial_query)
            else:
                follow_up_context = f"Original Query: {initial_query} \n\n Previous Report: {consensus_result}"
                queries = self._generate_expert_query(follow_up_context)

            retrieved_docs = self._retrieve_queries(queries)
            if retrieved_docs:
                self._update_knowledge(retrieved_docs, initial_query)
            expert_analyses = self._expert_analysis(self.current_knowledge, initial_query)
#            expert_analyses.append(self._cot_generalmedicine(initial_query))
            consensus_result = self._check_consensus(initial_query, expert_analyses, )

            self.iteration_history.append({
                'iteration': iteration,
                'queries': queries,
                'docs': retrieved_docs,
                'analyses': expert_analyses,
                'consensus': consensus_result
            })

            if "ensus: yes" in consensus_result.lower():
                return self._final_answer_pick(consensus_result), self.iteration_history
            iteration += 1

        return self._final_answer_pick(consensus_result), self.iteration_history

    def _update_knowledge(self, new_documents: List[str], original_query: str):
        """Update the current knowledge base with new documents."""
        combined_docs = "\n".join(set(new_documents) - set(self.current_knowledge))
        self.current_knowledge.append(combined_docs)

    def _summarize_documents(self, documents: str, original_query: str) -> str:
        summary_prompt = f"""Create a document that contains key information from the following set of medical documents, considering their relevance to the original query:

        Original Query: {original_query}

        Documents: {documents}

        Please provide a medical document with clinical information that might be helpful for answering the query. Do not suggest or conclude any answer choice."""
        try:
            return self._clean_text(self._call_llm([
                {"role": "system", "content": "You are a medical document writer including key information from documents."},
                {"role": "user", "content": summary_prompt}
            ]))
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
        ])
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
            ]))

            if "ield: " in response:
                domain_list = [domain.strip() for domain in response.split("ield: ")[1].split('|') if domain.strip()]
                if len(domain_list) == 1:
                    domain_list.append('General Medicine')
                return domain_list
            raise ValueError("Delimiter 'ield: ' not found in the response.")
        except (IndexError, ValueError, Exception) as e:
            return ['General Medicine'] * num_fields

    def _generate_expert_query(self, context: str) -> List[str]:
        query_prompt = """Write a medical passage that can help answer the given query. Include key information or terminology for the answer.
...
Example:

Query: A 39-year-old woman presents to the family medicine clinic to be evaluated by her physician for weight gain. She reports feeling fatigued most of the day despite eating a healthy diet and exercising regularly. The patient smokes a half-pack of cigarettes daily and has done so for the last 23 years. She is employed as a phlebotomist by the Red Cross. She has a history of hyperlipidemia for which she takes atorvastatin. She is unaware of her vaccination history, and there is no documented record of her receiving any vaccinations. Her heart rate is 76/min, respiratory rate is 14/min, temperature is 37.3°C (99.1°F), body mass index (BMI) is 33 kg/m2, and blood pressure is 128/78 mm Hg. The patient appears alert and oriented. Lung and heart auscultation are without audible abnormalities. The physician orders a thyroid panel to determine if that patient has hypothyroidism. Which of the following recommendations may be appropriate for the patient at this time? A) Hepatitis B vaccination B) Low-dose chest CT C) Hepatitis C vaccination D) Shingles vaccination
Passage: against vaccine-preventable diseases. Every visit by an adult to a health-care provider should be an opportunity to provide this protection. Several factors need to be con sidered before any patient is vaccinated. These include the susceptibility of the patient, the risk of exposure to the disease, the risk from the disease, and the benefits and risks from the immunizing agent. Physicians should maintain detailed information about previous vaccina tions received by each individual, including type of vaccination, date of receipt, and adverse events, if any, following vaccination. Information should also include the person's history of vaccine-preventable illnesses, occupation, and lifestyle. Vaccine histories ideally should be based on written documentation to ascertain whether vaccines and toxoids were administered at appropriate ages and at proper intervals. Close attention to factors

Query: A 23-year-old male presents to his primary care physician after an injury during a rugby game. The patient states that he was tackled and ever since then has had pain in his knee. The patient has tried NSAIDs and ice to no avail. The patient has no past medical history and is currently taking a multivitamin, fish oil, and a whey protein supplement. On physical exam you note a knee that is heavily bruised. It is painful for the patient to bear weight on the knee, and passive motion of the knee elicits some pain. There is laxity at the knee to varus stress. The patient is wondering when he can return to athletics. Which of the following is the most likely diagnosis? A) Medial collateral ligament tear B) Lateral collateral ligament tear C) Anterior cruciate ligament tear D) Posterior cruciate ligament tear
Passage: Diagnosing PCL Injuries: History, Physical Examination, Imaging Studies, Arthroscopic Evaluation. Isolated posterior cruciate ligament (PCL) injuries are uncommon and can be easily missed with physical examination. The purpose of this article is to give an overview of the clinical, diagnostic and arthroscopic evaluation of a PCL injured knee. There are some specific injury mechanisms that can cause a PCL including the dashboard direct anterior blow and hyperflexion mechanisms. During the diagnostic process it is important to distinguish between an isolated or multiligament injury and whether the problem is acute or chronic. Physical examination can be difficult in an acutely injured knee because of pain and swelling, but there are specific functional tests that can indicate a PCL tear. Standard x-ray's and stress views are very useful imaging modalities

Query: A 4-year-old male is accompanied by his mother to the pediatrician. His mother reports that over the past two weeks, the child has had intermittent low grade fevers and has been more lethargic than usual. The child’s past medical history is notable for myelomeningocele complicated by lower extremity weakness as well as bowel and bladder dysfunction. He has been hospitalized multiple times at an outside facility for recurrent urinary tract infections. The child is in the 15th percentile for both height and weight. His temperature is 100.7°F (38.2°C), blood pressure is 115/70 mmHg, pulse is 115/min, and respirations are 20/min. Physical examination is notable for costovertebral angle tenderness that is worse on the right. Which of the following would most likely be found on biopsy of this patient’s kidney? A) Mononuclear and eosinophilic infiltrate B) Replacement of renal parenchyma with foamy histiocytes C) Destruction of the proximal tubule and medullary thick ascending limb D) Tubular colloid casts with diffuse lymphoplasmacytic infiltrate
Passage: The natural history of urinary infection in adults. The vast majority of otherwise healthy adults with anatomically and functionally normal urinary tracts experience few untoward long-term consequences from symptomatic or asymptomatic UTIs. Effective early treatment of symptomatic infection rapidly curtails bacterial invasion and the resulting inflammatory response. Rarely, uncomplicated acute pyelonephritis causes suppuration and renal scarring. Urinary infections in patients with renal calculi, obstructed urinary tract, neurogenic bladder, or diabetes are frequently much more destructive and have ongoing sequelae. Strategies to treat both the infection and the complications are often necessary to alter this outcome.
...

Query: {}
Passage:"""
        all_queries = []
        for role in self.expert_roles:
            try:
                response = self._clean_text(self._call_llm([
                    {"role": "system", "content": f"You are a medical expert in {role}."},
                    {"role": "user", "content": query_prompt.format(context)}
                ]))
                all_queries.append(response)
            except Exception as e:
                print(f"Error generating queries for {role}: {str(e)}")
        return all_queries[:]

    def _expert_analysis(self, documents_or_knowledgebase: List[str], query: str) -> List[Dict]:
        """Perform expert analysis on the current knowledge base."""
        analysis_prompt = f"""Analyze this knowledge base and solve the query:

        Current Knowledge: {self._format_docs_for_prompt(documents_or_knowledgebase)[:self.doc_length]}

        Original Query: {query}

        Provide:
        1. Key Information: <Critical information>
        2. Remaining Questions: <Gaps to address>
        3. Reasoning: <Justification>
        4. Answer: <You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\", """

        return [{
            'role': role,
            'analysis': self._clean_text(self._call_llm([
                {"role": "system", "content": f"You are a {role} specialist analyzing medical information."},
                {"role": "user", "content": analysis_prompt}
            ]))
        } for role in self.expert_roles]

    def _cot_generalmedicine(self, query: str) -> Dict:
        return {
            'role': "General Medicine",
            'analysis': self._clean_text(self._call_llm([
                {"role": "system", "content": "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_id]) [answer_string]\""},
                {"role": "user", "content": f"Question: {query}"}
            ]))
        }

    def _retrieve_queries(self, queries: List[str]) -> List[str]:
        retrieved_docs = []
        for query in queries:
            docs = utils.rerank(query, utils.retrieve_filtered_sources(query, retrieval_client, topk = self.retrieve_topk))
            retrieved_docs.extend(docs[:self.rerank_topk])
        seen = set()
        seen_add = seen.add
        return [x for x in retrieved_docs if not (x in seen or seen_add(x))]

    def _check_consensus(self, query, expert_analyses: List[Dict]) -> str:
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
            {"role": "system", "content": "You are evaluating expert opinions for consensus on a question."},
            {"role": "user", "content": consensus_prompt.format(
                a='\n\n'.join([f"Expert ({a['role']}):\n{a['analysis']}" for a in expert_analyses]),
                q=query
            )}
        ]))

        return response

    def _identify_knowledge_gaps(self) -> str:
        """Identify knowledge gaps from the last iteration."""
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

    def _final_answer_pick(self, text):
        return self._call_llm([
                    {"role": "system", "content": f"You are a answer parser. Pick an answer even if there is no consensus."},
                    {"role": "user", "content": f"Only output A, B, C, or D from {text}"}
                ])

medqa_test = []
with open("datasets/MedQA/50_sampled_hard_medqa/test.jsonl", 'r') as jsfile:
    for line in jsfile:
        medqa_test.append(json.loads(line))

medqa_test = medqa_test[start_idx:start_idx+sample_size]

queries = [f"{test['question']}\n\nOptions: (A) {test['options']['A']} (B) {test['options']['B']} (C) {test['options']['C']} (D) {test['options']['D']}" for test in medqa_test]
print(queries[0])
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
