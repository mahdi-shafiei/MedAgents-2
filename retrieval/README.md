## 🛠️ Setup the retriever
1. 🐳 Install Docker if you haven't already.
   - For installation instructions, visit: [Get Docker](https://docs.docker.com/get-started/get-docker/)
2. 📦 Install the pymilvus package:
```bash
pip install -U pymilvus
```
3. ⬇️ Download and run the Milvus standalone installation script:
```bash
# Download the installation script
$ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
# Start the Docker container
$ bash standalone_embed.sh start
```
## 📤 Upload the corpus
This section describes how to upload your document corpus to Milvus. The corpus should be organized in the following structure under `medagents/retrieval/corpus/{corpus_name}/`:
- 📄 `data/`: Contains the raw text chunks
- 🔢 `vector/`: Contains pre-computed embeddings for each chunk
- 📋 `json/`: Contains metadata for each chunk

The `upload_all.sh` script will upload multiple corpora (cpg, recop, textbooks, statpearls) to separate Milvus collections. Each collection will be named as `{corpus_name}_2`.

To upload all corpora:
```bash
bash upload_all.sh
```