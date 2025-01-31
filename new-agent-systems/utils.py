from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import os
from natsort import natsorted
import json
from tqdm import tqdm
import numpy as np
import regex as re

def initialize_models(device):
    model_q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
    tokenizer_q = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

    model_c = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(device)
    tokenizer_c = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")

    return model_q, tokenizer_q, model_c, tokenizer_c

def retrieve(query, client,topk=100):
    search_res = client.search(
    collection_name='rag2',
    data=[
        medcpt_query_embedding_function(query)
    ],  
    limit=topk,  
    search_params={"metric_type": "IP", "params": {}},  
    output_fields=["text", 'source'],  
    )
    evidence_list = [result["entity"]["text"] for result in search_res[0][:topk]]
    return evidence_list

def retrieve_filtered_sources(query, client, allowed_sources = ["source == 'PubMed'", "source == 'PMC'", "source == 'Textbook'", "source == 'CPG'", "source == 'statpearls'", "source == 'recop'", "source == 'textbooks'", "source == 'cpg'"], topk=100):
    evidence_list = []
    query_embedding = medcpt_query_embedding_function(query)
    for source in allowed_sources:
        search_res = client.search(
            collection_name='rag2',
            data=[query_embedding               
            ],  
            limit=topk,  
            search_params={"metric_type": "IP", "params": {}},  
            output_fields=["text", 'source'], 
            filter = source
            )
        evidence_list.extend([result["entity"]["text"] for result in search_res[0][:topk]])
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

medical_specialties_gpt_selected = [
    "Internal Medicine",
    "Emergency Medicine",
    "Infectious Disease",
    "Pediatrics",
    "Neurology",
    "Endocrinology",
    "Cardiology",
    "Pharmacology",
    "Pulmonology",
    "Gastroenterology",
    "Hematology",
    "Radiology",
    "Nephrology",
    "Psychiatry",
    "Oncology",
    "Genetics",
    "Dermatology",
    "Pathology",
    "Geriatrics",
    "Immunology",
    "Rheumatology",
    "Urology",
    "Obstetrics and Gynecology",
    "Surgery",
    "Preventive Medicine",
    "Critical Care Medicine",
    "Toxicology",
    "Anesthesiology",
    "Neurosurgery",
    "Family Medicine",
    "Vascular Medicine",
    "Ophthalmology",
    "Orthopedics",
    "Occupational Medicine",
    "Sports Medicine",
    "Public Health",
    "Clinical Research",
    "Sleep Medicine",
    "Allergy and Immunology",
    "Biostatistics",
    "Medical Ethics",
    "Neonatology",
    "Nutrition",
    "Epidemiology",
    "Gastroenterology",
    "Rehabilitation Medicine",
    "Sexual Health",
    "Reproductive Medicine",
    "Transplant Medicine",
    "Clinical Pathology"
]