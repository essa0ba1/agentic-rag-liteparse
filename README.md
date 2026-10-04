# Agentic RAG + LiteParse + LangGraph

A document retrieval-augmented generation (RAG) system with agentic reasoning, LiteParse document parsing, and LangGraph workflow orchestration.

## Features

- **Agentic RAG**: Uses LangGraph to implement an agentic retrieval workflow with automatic query rewriting, document grading, and iterative retrieval
- **LiteParse Integration**: High-quality document parsing for PDFs, Word documents, Excel, CSV, and more
- **Hybrid Search**: Combines dense ONNX embeddings with sparse BM25 for improved retrieval accuracy
- **ModernBERT Embeddings**: State-of-the-art embeddings with quantized (Q4) and unquantized model options
- **Qdrant Vector Database**: Efficient vector storage with hybrid search capabilities
- **Gradio Web UI**: User-friendly interface for document ingestion, management, and chat
- **Docker Support**: Containerized deployment with automated model downloads

## Architecture

### Agentic Workflow (LangGraph)

The retrieval system uses a LangGraph-based agentic workflow:

1. **Retrieve**: Search the vector database for relevant chunks
2. **Grade Documents**: Evaluate retrieved chunks for relevance to the query
3. **Rewrite Query**: If insufficient relevant chunks found, rewrite the query for better retrieval
4. **Generate**: Generate an answer using the relevant chunks with citations

The agent automatically retries retrieval with rewritten queries up to 3 times if initial results are insufficient.

### Document Processing Pipeline

1. **Parse**: Use LiteParse to extract text from various file formats
2. **Chunk**: Recursive chunking with configurable parameters
3. **Embed**: Generate embeddings using ModernBERT (quantized or full precision)
4. **Index**: Store chunks in Qdrant with dense + sparse vectors

## Installation

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Qdrant (runs via Docker Compose)

### Local Setup

1. Clone the repository and navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download embedding models:

Create the models directory and download the ONNX models from Hugging Face:

```bash
mkdir -p models/modernbert
cd models/modernbert

# Download quantized model (Q4, faster, lower memory)
wget https://huggingface.co/nomic-ai/modernbert-embed-base/resolve/main/onnx/model_q4.onnx

# Download unquantized model (full precision, better quality)
wget https://huggingface.co/nomic-ai/modernbert-embed-base/resolve/main/onnx/model.onnx

cd ../..
```

Alternatively, use the Hugging Face CLI:
```bash
pip install huggingface-hub
huggingface-cli download nomic-ai/modernbert-embed-base --local-dir models/modernbert --local-dir-use-symlinks False
```

4. Start Qdrant:
```bash
docker-compose up qdrant -d
```

5. Set up llama.cpp for the LLM:

Install llama.cpp:
```bash
# Ubuntu/Debian
sudo apt-get install llama-cpp

# Or build from source
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
sudo make install
```

Download a quantized LLM model (example with LFM2-1.2B):
```bash
# Create directory for GGUF models
mkdir -p ~/ggufs

# Download a quantized model (adjust URL as needed)
wget -O ~/ggufs/LFM2-1.2B-RAG-Q4_K_M.gguf <model-url>
```

Start the llama.cpp server:
```bash
llama serve -m ~/ggufs/LFM2-1.2B-RAG-Q4_K_M.gguf --jinja -c 8192 --port 8000
```

The server will start on `http://127.0.0.1:8000` by default.

6. Configure your LLM endpoint in `src/config.py`:
```python
LLM_BASE_URL = "http://127.0.0.1:8000"  # llama.cpp server
LLM_API_KEY = "x"  # llama.cpp doesn't require an API key
```

7. Run the Gradio UI:
```bash
python src/ui.py
```

Access the UI at `http://127.0.0.1:7060`

### Docker Deployment

Build and run with Docker Compose:

```bash
docker-compose up --build
```

The Docker image automatically downloads the ONNX embedding models during build.

## Usage

### Document Ingestion

1. Navigate to the **Ingest** tab
2. Upload files or paste text
3. Choose embedding model:
   - **Quantized (Q4)**: Faster, lower memory usage
   - **Unquantized**: Full precision, better quality
4. Click "Run pipeline: Parse → Chunk → Index"

Supported file formats:
- PDF, DOCX, DOC (via LiteParse)
- XLSX, XLS, CSV (via LiteParse)
- TXT, MD (plain text)

### Chat with Documents

1. Navigate to the **Chat** tab
2. Ask questions about your indexed documents
3. The agent will:
   - Retrieve relevant chunks
   - Grade them for relevance
   - Rewrite the query if needed
   - Generate an answer with citations

### Document Management

Use the **Documents** tab to:
- View all indexed documents
- Remove individual documents
- Delete uploaded files

### System Administration

The **System** tab provides:
- Connection status for Qdrant and LLM
- Current embedding model information
- Index management (create, check status, reset)

## Configuration

### Chunking

Edit `chunker_config.json` to customize chunking behavior:

```json
{
  "chunk_size": 512,
  "chunk_overlap": 50,
  "separator": "\n\n"
}
```

### Embedding Models

Choose between quantized and unquantized models in the UI or configure in `src/config.py`:

```python
EMBEDDING_MODEL_PATH_QUANTIZED = "models/modernbert/model_q4.onnx"
EMBEDDING_MODEL_PATH_UNQUANTIZED = "models/modernbert/model.onnx"
```

### Hybrid Search

Configure hybrid search parameters in `src/config.py`:

```python
USE_HYBRID_SEARCH = True
SPARSE_MODEL_NAME = "Qdrant/bm25"
HYBRID_PREFETCH_MULTIPLIER = 4
TOP_K = 5
```

## Project Structure

```
.
├── src/
│   ├── agent.py              # LangGraph agentic workflow
│   ├── chunking.py           # Document chunking logic
│   ├── config.py             # Configuration settings
│   ├── data_pipeline.py      # Ingestion and indexing pipeline
│   ├── embedding.py          # ONNX embedding implementation
│   ├── hybrid_store.py       # Qdrant hybrid search store
│   ├── llm.py                # LLM client
│   ├── parsing.py            # LiteParse integration
│   ├── retriever.py          # Retrieval tools for agent
│   ├── ui.py                 # Gradio web interface
│   └── ...
├── data/                     # Data directory (uploads, logs)
├── models/                   # ONNX embedding models
├── config/                   # Configuration files
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker Compose orchestration
└── requirements.txt         # Python dependencies
```

## Advanced Features

### Retrieval Evaluation

Run retrieval evaluation to assess system performance:

```bash
python src/run_retrieval_eval.py
```

### Custom Metadata

Add custom metadata during ingestion using the CLI:

```bash
python -m src.data_pipeline document.pdf --meta category=research --tags important
```

### Logging

Retrieval operations are logged to `data/retrieval_chunks.jsonl` for audit trails and analysis.

## Model Switching

When switching between quantized and unquantized models:

1. Reset the index in the System tab
2. Re-index all documents with the new model choice
3. The system automatically uses the same model for retrieval

**Note**: Using different models for indexing and retrieval can reduce accuracy.

## Troubleshooting

### Port Conflicts

If port 8080 or 7060 is in use, modify the ports in `docker-compose.yml` or `src/config.py`.

### Model Loading Issues

Ensure the ONNX models are in `models/modernbert/`:
- `model.onnx` (unquantized)
- `model_q4.onnx` (quantized)

### Qdrant Connection

Verify Qdrant is running:
```bash
docker-compose ps qdrant
```

## License

[Your License Here]

## Acknowledgments

- [LiteParse](https://github.com/slashmarkai/liteparse) for document parsing
- [LangGraph](https://github.com/langchain-ai/langgraph) for agentic workflow orchestration
- [Qdrant](https://qdrant.tech/) for vector database
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) for embeddings
- [Gradio](https://gradio.app/) for the web UI
