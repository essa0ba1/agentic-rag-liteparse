EMBEDDING_MODEL_PATH_QUANTIZED = "models/modernbert/model_q4.onnx"
EMBEDDING_MODEL_PATH_UNQUANTIZED = "models/modernbert/model.onnx"
EMBEDDING_MODEL_PATH = EMBEDDING_MODEL_PATH_QUANTIZED  # Default to quantized
EMBEDDING_TOKENIZER_NAME = "answerdotai/ModernBERT-base"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "my_documents"
TOP_K = 5
CHUNKER_CONFIG_PATH = "chunker_config.json"

# Log every retrieval (console + optional JSONL audit trail)
RETRIEVAL_LOG_PATH = "data/retrieval_chunks.jsonl"
RETRIEVAL_LOG_PREVIEW_CHARS = 400

# Hybrid search (dense ONNX + sparse BM25, fused with RRF)
USE_HYBRID_SEARCH = True
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
HYBRID_PREFETCH_MULTIPLIER = 4

LLM_BASE_URL = "http://127.0.0.1:8000"
LLM_API_KEY = "x"
