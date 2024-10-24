import retrieve_utils
import argparse
import torch
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

if not client.has_collection(collection_name="medagents"):
    retrieve_utils.data_upload(client, collection_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, nargs='+', default=[0], help="CUDA device numbers, e.g., --cuda 0 1 for cuda:0 and cuda:1")
    args = parser.parse_args()
    
    devices = [i for i in args.cuda if torch.cuda.is_available() and i < torch.cuda.device_count()]
    query_encoder = retrieve_utils.MedCPTQueryEncoder(devices=devices)
    
    query_embedding = query_encoder.encode('A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?')

    search_res = client.search(
        collection_name='medagents',
        data=[query_embedding
        ],  
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )
    print(search_res)