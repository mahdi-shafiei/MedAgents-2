import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import os
from natsort import natsorted
import json
from tqdm import tqdm
import numpy as np
import regex as re
from pymilvus import MilvusClient, DataType

client = MilvusClient(
    uri="http://localhost:19530"
)


def data_upload(client, collection_name, vector_paths, text_paths, batch_size=10000):
    global_offset = 0

    if not client.has_collection(collection_name):
        client.create_collection(
            collection_name=collection_name,
            dimension=768,
            metric_type="IP",  
            consistency_level="Strong",
        )

    print(f"Uploading data to '{collection_name}' collection in Milvus...\n")

    for vector_path, text_path in zip(vector_paths, text_paths):
        print(f"Processing files:\nVector: {vector_path}\nText: {text_path}")

        vectors = np.load(vector_path)
        with open(text_path, 'r') as jsfile:
            texts = json.load(jsfile)

        data_to_insert = [
            {"vector": vector.tolist(), "text": text, "source": os.path.basename(text_path).split("_")[0]}
            for vector, text in zip(vectors, texts)
        ]

        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i+batch_size]
            for idx, item in enumerate(batch):
                if len(item['text']) > 20000:
                    item['text'] = item['text'][:20000]
                item['id'] = global_offset + idx
            client.insert(collection_name=collection_name, data=batch)
            global_offset += len(batch)

        print(f"Uploaded data from {text_path}!")

vector_paths = [
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_0.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_1.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_2.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_3.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_4.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_5.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_6.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_7.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_8.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_9.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_10.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_11.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_12.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_13.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_14.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_15.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_16.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_17.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_18.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_19.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_20.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_21.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_22.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_23.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_24.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_25.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_26.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_27.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_28.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_29.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_30.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_31.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_32.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_33.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_34.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_35.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_36.npy',
'/data/jiwoong/retriever/embeddings/pubmed/PubMed_Embeds_37.npy',
'/data/jiwoong/retriever/embeddings/textbook/Textbook_Total_Embeds.npy',
'/data/jiwoong/retriever/embeddings/pmc/PMC_Abs_Embeds.npy',
'/data/jiwoong/retriever/embeddings/pmc/PMC_Main_Embeds.npy',
'/data/jiwoong/retriever/embeddings/cpg/CPG_Total_Embeds.npy',
]

text_paths = [
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_0.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_1.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_2.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_3.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_4.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_5.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_6.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_7.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_8.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_9.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_10.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_11.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_12.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_13.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_14.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_15.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_16.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_17.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_18.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_19.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_20.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_21.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_22.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_23.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_24.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_25.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_26.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_27.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_28.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_29.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_30.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_31.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_32.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_33.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_34.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_35.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_36.json',
'/data/jiwoong/retriever/articles/pubmed/PubMed_Articles_37.json',
'/data/jiwoong/retriever/articles/textbook/Textbook_Total_Articles.json',
'/data/jiwoong/retriever/articles/pmc/PMC_Main_Articles.json',
'/data/jiwoong/retriever/articles/pmc/PMC_Abs_Articles.json',
'/data/jiwoong/retriever/articles/cpg/CPG_Total_Articles.json',
]

data_upload(client, collection_name='rag2', vector_paths=vector_paths, text_paths=text_paths)
