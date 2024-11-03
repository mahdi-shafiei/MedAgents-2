import os
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from pymilvus import MilvusClient
from dotenv import load_dotenv
import utils
import re

load_dotenv()

retrieval_client = MilvusClient(uri="http://localhost:19530")
llm_client = AzureOpenAI(
    azure_endpoint="https://azure-openai-miblab-ncu.openai.azure.com/",
    api_key=os.getenv("azure_api_key"),
    api_version="2024-08-01-preview"
)

class CIDER:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.current_knowledge = []
        self.iteration_history = []
        self.expert_roles = []
        self.retrieve_topk = 5
        self.rerank_topk = 10

    def _clean_text(self, text: str) -> str:
        """Remove markdown formatting and clean text."""
        if not isinstance(text, str):
            return ""
        return text.replace('**', '').replace("\'\'\'", '').strip()

    def _call_llm(self, messages: List[Dict]) -> str:
        """Make an API call to the LLM and return the response."""
        response = llm_client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=2048
        )
#        print("\n" + "="*50 + " LLM Input Messages " + "="*50)
#        for message in messages:
#            print(f"{message['role']}: {message['content']}\n")
        output = response.choices[0].message.content
#        print("\n" + "="*50 + " LLM Output Response " + "="*50)
#        print(output + "\n")
        return self._clean_text(output)


    def process_query(self, initial_query: str) -> Dict[str, Any]:
        """Main query processing loop."""
        iteration = 0
        max_iterations = 5

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
            #expert_analyses.append(self._cot_generalmedicine(initial_query))
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
        if combined_docs.strip():
            summary = self._summarize_documents(combined_docs, original_query)
            if summary:
                self.current_knowledge.append(summary)

    def _summarize_documents(self, documents: str, original_query: str) -> str:
        summary_prompt = f"""Summarize the key insights from the following set of medical documents, considering their relevance to the original query:

        Original Query: {original_query}

        Documents: {documents}

        Please provide a concise and objective summary of the most clinically relevant information. Use exact phrases from the documents where appropriate, and do not suggest or conclude any answer choice. Focus on explaining relevant mechanisms, key facts, and insights that address the query."""
        try:
            return self._clean_text(self._call_llm([
                {"role": "system", "content": "You are a medical summarizer extracting key insights from documents."},
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
        """Generate expert queries based on context."""
        query_prompt = """Generate up to three specific medical queries addressing:

        {context}

        Consider:
        1. Expert disagreements
        2. Additional information needed
        3. Remaining knowledge gaps

        Format:
        1st Query: <Primary concerns>
        2nd Query: <Secondary aspects>
        3rd Query: <Remaining gaps>

        Make queries specific and targeted."""

        all_queries = []
        for role in self.expert_roles:
            try:
                response = self._clean_text(self._call_llm([
                    {"role": "system", "content": f"You are a medical expert in {role}."},
                    {"role": "user", "content": query_prompt.format(context=context)}
                ]))
                queries = [
                    line.split(":")[-1].strip() for line in response.split("\n")
                    if line.strip().startswith(("1st Query", "2nd Query", "3rd Query"))
                ]
                all_queries.extend(queries)
            except Exception as e:
                print(f"Error generating queries for {role}: {str(e)}")
        return all_queries[:]

    def _expert_analysis(self, documents_or_knowledgebase: List[str], query: str) -> List[Dict]:
        """Perform expert analysis on the current knowledge base."""
        analysis_prompt = f"""Analyze this knowledge base and solve the query:

        Current Knowledge: {self._format_docs_for_prompt(documents_or_knowledgebase)}

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
            docs = utils.rerank(query, utils.retrieve(query, retrieval_client, topk = self.retrieve_topk))
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
