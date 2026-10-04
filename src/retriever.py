import logging

from langchain_core.tools import tool
from qdrant_client.http import models
from transformers import AutoTokenizer

import config
from data_pipeline import load_model_config
from embedding import ONNXEmbeddings
from hybrid_store import HybridQdrantStore
from metadata import format_hit_for_llm, log_retrieved_hits

logger = logging.getLogger(__name__)

_tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_TOKENIZER_NAME)

# Load the model choice that was used for indexing
use_quantized = load_model_config()
model_path = config.EMBEDDING_MODEL_PATH_QUANTIZED if use_quantized else config.EMBEDDING_MODEL_PATH_UNQUANTIZED
logger.info(f"Loading embedding model: {model_path}")

_embeddings = ONNXEmbeddings(
    model_path,
    _tokenizer,
    providers=["CPUExecutionProvider"],
)

_store = HybridQdrantStore(_embeddings)


def _build_filter(
    source: str | None = None,
    file_type: str | None = None,
) -> models.Filter | None:
    conditions: list[models.FieldCondition] = []
    if source:
        conditions.append(
            models.FieldCondition(key="source", match=models.MatchValue(value=source)),
        )
    if file_type:
        conditions.append(
            models.FieldCondition(key="file_type", match=models.MatchValue(value=file_type)),
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)



def search_qdrant(
    query: str,
    source: str | None = None,
    file_type: str | None = None,
) -> list[str] | str:
    """
    Search the user's indexed knowledge base (hybrid dense + keyword/BM25).

    Use this tool when answering a question that requires information
    from the user's documents, uploaded files, or indexed knowledge.

    Do not use this tool for general knowledge questions that can be
    answered without consulting the user's knowledge base.

    Args:
        query: The specific information to search for.
        source: Optional exact filename filter (e.g. "report.pdf").
        file_type: Optional file extension filter (e.g. "pdf", "md").

    Returns:
        A list of formatted passages with their source metadata, or a message
        string indicating that no relevant information was found.
    """
    query_filter = _build_filter(source=source, file_type=file_type)
    results = _store.search(query=query, limit=config.TOP_K, query_filter=query_filter)
    log_retrieved_hits(
        query,
        results,
        log_path=config.RETRIEVAL_LOG_PATH or None,
        preview_chars=config.RETRIEVAL_LOG_PREVIEW_CHARS,
        logger=logger,
    )
    if not results:
        return "No relevant information found in the knowledge base."

    return [format_hit_for_llm(hit, i) for i, hit in enumerate(results, start=1)]
