from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import os
from natsort import natsorted
import json
from tqdm import tqdm
import numpy as np
import regex as re

class MedCPTRetriever:
    def __init__(self, device="cpu"):
        self.device = device
        self.model_q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
        self.tokenizer_q = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        
        self.model_c = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(device)
        self.tokenizer_c = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")

    def retrieve(self, query, client, topk=100):
        search_res = client.search(
            collection_name='rag2',
            data=[
                self._medcpt_query_embedding_function(query)
            ],  
            limit=topk,  
            search_params={"metric_type": "IP", "params": {}},  
            output_fields=["text", 'source'],  
        )
        evidence_list = [result["entity"]["text"] for result in search_res[0][:topk]]
        return evidence_list

    def retrieve_filtered_sources(self, query, client, allowed_sources=["source == 'PubMed'", "source == 'PMC'", "source == 'Textbook'", "source == 'CPG'", "source == 'statpearls'", "source == 'recop'", "source == 'textbooks'", "source == 'cpg'"], topk=100):
        evidence_list = []
        query_embedding = self._medcpt_query_embedding_function(query)
        for source in allowed_sources:
            search_res = client.search(
                collection_name='rag2',
                data=[query_embedding],  
                limit=topk,  
                search_params={"metric_type": "IP", "params": {}},  
                output_fields=["text", 'source'], 
                filter=source
            )
            evidence_list.extend([result["entity"]["text"] for result in search_res[0][:topk]])
        return evidence_list

    def _medcpt_query_embedding_function(self, docs):
        encoded = self.tokenizer_q(
            docs,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            embeds = self.model_q(**encoded).last_hidden_state[:, 0, :]
        embeds = embeds.cpu().numpy()
        return embeds[0].tolist()

    def rerank(self, query, doc_list):
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
