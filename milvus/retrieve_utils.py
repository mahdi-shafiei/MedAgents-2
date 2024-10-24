import json
import torch
from tqdm.auto import tqdm
from natsort import natsorted
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

def data_upload(client, collection_name):
    client.create_collection(
        collection_name="medagents",
        dimension=768,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
    )
    print(f"data uploading to '{collection_name}' collection in milvus client...\n")
    corpus_list = [recop]
    for corpus in corpus_list:
        print(f"Processing corpus: {corpus}")
        file_list = natsorted([os.path.join(f"corpus/{corpus}/data/", f) for f in os.listdir(f"corpus/{corpus}/data/")])
        for file_path in file_list:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            for i in range(0, len(data), 10000):
                client.insert(collection_name=collection_name, data=data[i:i+10000])
            print(f"{file_path} uploaded to '{collection_name}' collection in Milvus!\n")
        print(f"{corpus} Uploaded!")

class MedCPTQueryEncoder:
    def __init__(self, devices, model_name="ncbi/MedCPT-Query-Encoder"):
        self.device = f"cuda:{devices[0]}"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, docs):
        encoded = self.tokenizer(
            docs,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            embeds = self.model(**encoded).last_hidden_state[:, 0, :]
        embeds = embeds.cpu().numpy()
        return embeds[0].tolist()