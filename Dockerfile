# ────────────────────────────────────────────────────────────────────────────
#  Dockerfile — Document RAG pipeline (Gradio + ONNX embeddings + Qdrant)
# ────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed by some Python packages (build wheels / native libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a dedicated prefix so we can copy them
# cleanly into the final image.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="document-rag"
LABEL org.opencontainers.image.description="Gradio RAG pipeline: parse → chunk → embed → hybrid search → chat"

WORKDIR /app

# Runtime system libraries (libgomp for onnxruntime, wget for model downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY src/        ./src/
COPY config/     ./config/
COPY data/       ./data/
COPY chunker_config.json .

# Create models directory and download ONNX models from Hugging Face
RUN mkdir -p models/modernbert && \
    cd models/modernbert && \
    wget -O model.onnx https://huggingface.co/nomic-ai/modernbert-embed-base/resolve/main/onnx/model.onnx && \
    wget -O model_q4.onnx https://huggingface.co/nomic-ai/modernbert-embed-base/resolve/main/onnx/model_q4.onnx

# Gradio web UI port
EXPOSE 7060

# Environment defaults — override at runtime with -e or docker-compose
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7060"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Default command: launch the Gradio UI
CMD ["python", "src/ui.py"]
