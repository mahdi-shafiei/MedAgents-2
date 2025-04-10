from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import os
from natsort import natsorted
import json
from tqdm import tqdm
import numpy as np
import regex as re
from typing import List, Dict, Tuple, Any, Optional, Union


def calculate_query_similarity(current_embedding: List[float], previous_embeddings: List[List[float]]) -> Tuple[float, int]:
    """
    Calculate cosine similarity between current embedding and a list of previous embeddings.
    
    Args:
        current_embedding: Embedding vector of the current query
        previous_embeddings: List of embedding vectors from previous queries
        
    Returns:
        Tuple containing the maximum similarity score and the index of the most similar embedding
    """
    current_tensor = torch.tensor(current_embedding).unsqueeze(0)
    previous_tensor = torch.tensor(previous_embeddings)
    
    similarities = torch.nn.functional.cosine_similarity(
        current_tensor, 
        previous_tensor,
        dim=1 
    )
    max_similarity = torch.max(similarities).item()
    max_idx = torch.argmax(similarities).item()
    return max_similarity, max_idx

class MedCPTRetriever:
    def __init__(self, device: str = "cpu"):
        """
        Initialize the MedCPT retriever with query encoder and cross-encoder models.
        
        Args:
            device: Device to run the models on (e.g., "cpu", "cuda")
        """
        self.device = device
        self.model_q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
        self.tokenizer_q = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        
        self.model_c = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(device)
        self.tokenizer_c = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")

    def retrieve(self, query: str, client: Any, topk: int = 100) -> List[str]:
        """
        Retrieve documents from a vector database using the query.
        
        Args:
            query: The search query
            client: Vector database client
            topk: Number of top documents to retrieve
            
        Returns:
            List of retrieved document texts
        """
        search_res = client.search(
            collection_name='rag2',
            data=[
                self.encode(query)
            ],  
            limit=topk,  
            search_params={"metric_type": "IP", "params": {}},  
            output_fields=["text", 'source'],  
        )
        evidence_list = [result["entity"]["text"] for result in search_res[0][:topk]]
        return evidence_list

    def retrieve_filtered_sources(
        self, 
        query: str, 
        client: Any, 
        allowed_sources: List[str] = ['cpg', 'recop', 'textbooks', 'statpearls'],
        topk: int = 100
    ) -> List[str]:
        """
        Retrieve documents from specific sources in the vector database.
        
        Args:
            query: The search query
            client: Vector database client
            allowed_sources: List of source collections to search in
            topk: Number of top documents to retrieve per source
            
        Returns:
            List of retrieved document texts with duplicates removed
        """
        evidence_list = []
        query_embedding = self.encode(query)
        for source in allowed_sources:
            search_res = client.search(
                collection_name=source,
                data=[query_embedding],  
                limit=topk,  
                search_params={"metric_type": "IP", "params": {}},  
                output_fields=["text", 'source'], 
            )
            evidence_list.extend([result["entity"]["text"] for result in search_res[0][:topk]])
        evidence_list = list(dict.fromkeys(evidence_list))
        return evidence_list

    def encode(self, query: str) -> List[float]:
        """
        Encode a query into an embedding vector using the query encoder.
        
        Args:
            query: The text query to encode
            
        Returns:
            List of floats representing the query embedding
        """
        encoded = self.tokenizer_q(
            query,
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

    def rerank(self, query: str, doc_list: List[str]) -> List[str]:
        """
        Rerank a list of documents based on relevance to the query using the cross-encoder.
        
        Args:
            query: The search query
            doc_list: List of documents to rerank
            
        Returns:
            List of documents reordered by relevance
        """
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
