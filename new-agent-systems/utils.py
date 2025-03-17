from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import os
from natsort import natsorted
import json
from tqdm import tqdm
import numpy as np
import regex as re
from logging import getLogger

logger = getLogger(__name__)

class MedicalRetriever:
    def __init__(self, device="cpu"):
        self.device = device
        
        # Load models
        self.model_q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(self.device)
        self.tokenizer_q = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.model_c = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(self.device)
        self.tokenizer_c = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        
        # Track OOM errors
        self.oom_count = 0
    
    def get_oom_count(self):
        return self.oom_count
        
    def query_embedding_function(self, docs):
        encoded = self.tokenizer_q(docs, truncation=True, padding=True, return_tensors='pt', max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            embeds = self.model_q(**encoded).last_hidden_state[:, 0, :]
            if next(self.model_q.parameters()).is_cuda:
                embeds = embeds.cpu()
            embeds = embeds.numpy()
        return embeds[0].tolist()
    
    def retrieve_only(self, query, retrieval_client, allowed_sources=['cpg', 'statpearls', 'recop', 'textbooks'], topk=100):
        evidence_list = []
        query_embedding = self.query_embedding_function(query)
        for source in allowed_sources:
            search_res = retrieval_client.search(
                collection_name=source,
                data=[query_embedding],
                limit=topk,  
                search_params={"metric_type": "IP", "params": {}},  
                output_fields=["text", 'source'], 
            )
            evidence_list.extend([result["entity"]["text"] for result in search_res[0][:topk]])
        return evidence_list
    
    def rerank_only(self, query, doc_list):
        pairs = [[query, doc] for doc in doc_list]
        with torch.no_grad():
            encoded = self.tokenizer_c(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()} 
            logits = self.model_c(**encoded).logits.squeeze(dim=1).detach().cpu()
        sorted_indices = torch.argsort(logits, descending=True)
        ranked_docs = [doc_list[i] for i in sorted_indices]
        return ranked_docs
    
    def retrieve(self, query, retrieval_client, retrieve_topk, rerank_topk):
        retrieved_docs = self.retrieve_only(query, retrieval_client, topk=retrieve_topk)
        logger.info(f"Retrieved docs: {retrieved_docs}")
        reranked_docs = self.rerank_only(query, retrieved_docs)
        logger.info(f"Reranked docs: {reranked_docs}")
        seen = set()
        unique_docs = []
        for doc in reranked_docs[:]:
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)
        return unique_docs[:rerank_topk]

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