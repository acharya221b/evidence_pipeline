import os

# Model Configs
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
CROSS_ENCODER_MODEL = "ncbi/MedCPT-Cross-Encoder"

# Data Paths
DATA_DIR = "."
DEF_INDEX_FILE = "/your-path/def_faiss.index"
DEF_META_FILE  = "/your-path/def_meta.filtered.json"
SEM_INDEX_FILE = "/your-path/semantic_faiss.index"
SEM_META_JSONL = "/your-path/semantic_nodes.filtered.jsonl"

# NebulaGraph Credentials
NEBULA_HOST = "127.0.0.1"
NEBULA_PORT = 9669
NEBULA_USER = "root"
NEBULA_PASSWORD = "your-password"  # Replace with your NebulaGraph password
NEBULA_SPACE = "schema_space"  # Replace with your NebulaGraph space name

# Ollama API Configs
OPENAI_BASE_URL = "https://your-ollama-instance.api"  # Replace with your Ollama instance URL
API_KEY = "sk-***"