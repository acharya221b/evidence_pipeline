import os

# Model Configs
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
CROSS_ENCODER_MODEL = "ncbi/MedCPT-Cross-Encoder"

# Data Paths
DATA_DIR = "."
DEF_INDEX_FILE = "/home/macharya/dev/graph_rag/def_faiss.index"
DEF_META_FILE  = "/home/macharya/dev/graph_rag/def_meta.filtered.json"
SEM_INDEX_FILE = "/home/macharya/dev/graph_rag/semantic_faiss.index"
SEM_META_JSONL = "/home/macharya/dev/graph_rag/semantic_nodes.filtered.jsonl"

# NebulaGraph Credentials
NEBULA_HOST = "127.0.0.1"
NEBULA_PORT = 9669
NEBULA_USER = "root"
NEBULA_PASSWORD = "nebula"
NEBULA_SPACE = "petagraph"

# Ollama API Configs
OPENAI_BASE_URL = "https://ollama.zib.de/api"
API_KEY = "sk-50837c3046664c90bc4367c9d5ebff3f"