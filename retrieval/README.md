## Setup the retriever

1. Install Docker if you haven't already.

2. Download and run the Milvus standalone installation script:

```bash
# Download the installation script
$ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start the Docker container
$ bash standalone_embed.sh start
```

3. Install the pymilvus package:

```bash
pip install pymilvus
```

## Upload the corpus

```bash
bash upload_all.sh
```
