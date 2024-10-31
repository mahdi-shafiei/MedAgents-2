from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import os
from natsort import natsorted
import json
from tqdm import tqdm
import numpy as np
import regex as re
from pymilvus import MilvusClient, DataType
from datetime import datetime
from openai import AzureOpenAI, OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

retrieval_client = MilvusClient(
    uri="http://localhost:19530"
)

llm_client = AzureOpenAI(
    azure_endpoint = "https://azure-openai-miblab-ncu.openai.azure.com/",
    api_key = os.getenv("azure_api_key"),
    api_version = "2024-08-01-preview",
)

#OPENAPI_API_KEY = 

#llm_client = OpenAI(
#    api_key=OPENAPI_API_KEY,
#)

#dataset_name = "hard_medqa"
#dataset_path = "/data/jiwoong/workspace/MedAgents/datasets/MedQA/50_sampled_hard_medqa/test.jsonl"
dataset_name = "medqa_oldrag"
dataset_path = "/data/jiwoong/workspace/benchmark/medqa/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl"
model_name = "gpt-4o-mini"
device = "cuda:0"
doc_stack_num = 100
temperature = 0
retrieved_documents_path = "/data/jiwoong/retriever/output/query2doc/evidence_medqa_gpt4omini_query2doc_fewshot.json"#"/data/jiwoong/retriever/output/query2doc/evidence_hard_medqa_gpt4omini_query2doc_fewshot.json"

output_save_dir = "/data/jiwoong/workspace/milvus/outputs"
formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
save_path = os.path.join(output_save_dir, dataset_name, model_name, formatted_time)
os.makedirs(save_path, exist_ok=True)

test_dataset = [] 
with open(dataset_path, 'r') as file:
    for line in file:
        test_dataset.append(json.loads(line))


model_q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
tokenizer_q = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")


model_c = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(device)
tokenizer_c = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")



retrieval_client = MilvusClient(
    uri="http://localhost:19530"
)


def generate_response(question):
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=question,
        logprobs=True,
        temperature=temperature,
        n=1
    )
    return response.choices[0].message.content

def generate_responses_in_parallel(prompts):
    result = [None] * len(prompts)  # Initialize list with None to store results in order
    # Function to generate response and store it at the correct index
    def generate_response_with_index(index, prompt):
        response = generate_response(prompt)
        return index, response
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks to the executor and keep track of the future objects
        futures = {executor.submit(generate_response_with_index, i, prompt): i for i, prompt in enumerate(prompts)}

        for future in tqdm(as_completed(futures), total=len(prompts)):
            index, response = future.result()  # Get the result when a thread completes
            result[index] = response  # Store the response in the correct index
    return result

def retrieve(query, source_filter , client, topk = 100):
    search_res = client.search(
    collection_name='rag2',
    data=[
        medcpt_query_embedding_function(query)
    ],  
    limit=topk,  
    search_params={"metric_type": "IP", "params": {}},  
    output_fields=["text", 'source'],  
    filter = source_filter 
    )
    
    evidence_list = [result["entity"]["text"] for result in search_res[0][:topk]]
    return evidence_list

def retrieve_filtered_sources(query, client, allowed_sources = ["source == 'PubMed'", "source == 'PMC'", "source == 'Textbook'", "source == 'CPG'", "source == 'statpearls'", "source == 'recop'", "source == 'textbooks'", "source == 'cpg'"], topk=100):
#def retrieve_filtered_sources(query, client, allowed_sources = ["source == 'statpearls'", "source == 'recop'", "source == 'textbooks'", "source == 'cpg'"], topk=100):
    evidence_list = []
    for source_filter  in allowed_sources:
        evidence_list.extend(retrieve(query, source_filter, client, topk=topk, ))
    return evidence_list


def medcpt_query_embedding_function(docs):
    encoded = tokenizer_q(
        docs,
        truncation=True,
        padding=True,
        return_tensors='pt',
        max_length=512,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        embeds = model_q(**encoded).last_hidden_state[:, 0, :]
    embeds = embeds.cpu().numpy()
    return embeds[0].tolist()

def rerank(query, doc_list):
    pairs = [[query, doc] for doc in doc_list]
    with torch.no_grad():
        encoded = tokenizer_c(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()} 
        logits = model_c(**encoded).logits.squeeze(dim=1).detach().cpu()
    sorted_indices = torch.argsort(logits, descending=True)
    ranked_docs = [doc_list[i] for i in sorted_indices]
    return ranked_docs

def encode_question_gen_query2doc(question: str,) -> list[str]:
    prompts= []
    few_shot_template = """...
Example:

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
Passage:
""".format(question)
    prompts.append({"role": "system", "content": "Write a medical passage that can help answer the given query. Include key information or terminology for the answer."})
    prompts.append({"role": "user", "content": few_shot_template})
    return prompts

def encode_question_gen(evidence: str,question: str,options: str,) -> list[str]:
    prompts= []
    prompt_rag = """Here is the relevant document:
{}

Here is the question:
{}

Here are the options:
{}

Please think step-by-step and generate your output.
""".format(evidence, question, options)
    prompts.append({"role": "system", "content": "The following is a multiple-choice question about medical knowledge. Solve this in a step-by-step fashion, starting by summarizing the available information. Output a single option from the given options as the final answer. You are strongly required to follow the specified output format; conclude your response with the phrase \"the answer is ([option_index]) [answer_string]\"."})
    prompts.append({"role": "user", "content": prompt_rag})
    return prompts

if not retrieved_documents_path:
    chatgpt_prompts = []
    for index, test in enumerate(test_dataset):
        chatgpt_prompts.append(encode_question_gen_query2doc(test['question'] + ' (A) ' + test['options']['A'] + ' (B) ' + test['options']['B'] + ' (C) ' + test['options']['C'] + ' (D) ' + test['options']['D'], ))

    pseudodocs = generate_responses_in_parallel(chatgpt_prompts)

    filename = "pseudodocs.json"
    with open(os.path.join(save_path, filename), 'w') as jsfile:
        json.dump(pseudodocs, jsfile)

    with open(os.path.join(output_save_dir,dataset_name,model_name,formatted_time,"pseudodocs.json"), 'w') as jsfile:
        json.dump(pseudodocs, jsfile)

    realdocs = []
    for pseudodoc in tqdm(pseudodocs):
        realdocs.append(retrieve(pseudodoc, retrieval_client, topk=100))
    #    realdocs.append(retrieve_filtered_sources(pseudodoc, client, allowed_sources = ["source == 'statpearls'", "source == 'recop'", "source == 'textbooks'", "source == 'cpg'"], topk=100))

    rankeddocs = []
    for index, realdoc in tqdm(enumerate(realdocs),total=len(test_dataset)):
        test = test_dataset[index]
        rankeddocs.append(rerank(
            test['question'] + ' (A) ' + test['options']['A'] + ' (B) ' + test['options']['B'] + ' (C) ' + test['options']['C'] + ' (D) ' + test['options']['D'],
            realdoc))
    filename = "rankeddocs.json"
    with open(os.path.join(save_path, filename), 'w') as jsfile:
        json.dump(rankeddocs, jsfile)

else:
    with open(retrieved_documents_path, 'r') as jsfile:
        rankeddocs = json.load(jsfile)

chatgpt_prompts = []
for index, test in enumerate(test_dataset):
    context = "\n".join(rankeddocs[index][:doc_stack_num])
    chatgpt_prompts.append(encode_question_gen(context, test['question'], '(A) ' + test['options']['A'] + ' (B) ' + test['options']['B'] + ' (C) ' + test['options']['C'] + ' (D) ' + test['options']['D'], ))

results = generate_responses_in_parallel(chatgpt_prompts)

temp = [0]*len(test_dataset)
for index, answer in enumerate(results):
    last_sentence = answer.split('answer is')[-1]
    matches = re.findall(r"\([A-Da-d]\)", last_sentence)
    chosen_option_2 = ""
    if matches:
        chosen_option_2 = matches[-1][1]
    if test_dataset[index]['answer_idx'] == chosen_option_2:
        temp[index] = 1
    else:
        temp[index] = 0
print(sum(temp) / len(temp))

filename = f"results_docstacknum={doc_stack_num}_temperature={temperature}_score={sum(temp)/len(temp):.4f}.json"
with open(os.path.join(save_path, filename), 'w') as jsfile:
    json.dump(results, jsfile)


