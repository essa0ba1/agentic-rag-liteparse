"""Ingest documents: parse → chunk → embed → Qdrant."""

from __future__ import annotations
from manifest import load_manifest, save_manifest, is_unchanged, file_hash, file_fingerprint
import argparse
import json
import logging
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from transformers import AutoTokenizer

import config
from chunking import DEFAULT_CHUNKER_CONFIG_PATH, chunk_file, ensure_chunker_config, load_chunker
from embedding import ONNXEmbeddings
from hybrid_store import (
    HybridQdrantStore,
    collection_status,
    delete_points_by_source,
    list_sources,
)
from metadata import (
    annotate_chunks,
    apply_document_metadata,
    build_file_metadata,
    parse_extra_metadata,
)
from parsing import file_to_document

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
MODEL_CONFIG_PATH = Path("data/model_config.json")


def save_model_config(use_quantized: bool) -> None:
    """Save the model choice used for indexing."""
    MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config_data = {"use_quantized": use_quantized}
    MODEL_CONFIG_PATH.write_text(json.dumps(config_data), encoding="utf-8")


def load_model_config() -> bool:
    """Load the model choice used for indexing (defaults to quantized)."""
    if not MODEL_CONFIG_PATH.exists():
        return True  # Default to quantized
    try:
        config_data = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
        return config_data.get("use_quantized", True)
    except Exception:
        return True  # Default to quantized on error


def build_store(use_quantized: bool = True) -> HybridQdrantStore:
    model_path = config.EMBEDDING_MODEL_PATH_QUANTIZED if use_quantized else config.EMBEDDING_MODEL_PATH_UNQUANTIZED
    logger.info(f"Loading embedding model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_TOKENIZER_NAME)
    embeddings = ONNXEmbeddings(
        model_path,
        tokenizer,
        providers=["CPUExecutionProvider"],
    )
    return HybridQdrantStore(embeddings)


# ── index management (plain client, no embedding model needed) ────────────────


def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


def index_status() -> dict[str, Any]:
    """Collection existence, point count, and per-document breakdown."""
    client = get_client()
    status = collection_status(client, config.COLLECTION_NAME)
    status["documents"] = list_sources(client, config.COLLECTION_NAME) if status["exists"] else []
    return status


def list_documents() -> list[dict[str, Any]]:
    """Documents currently in the index: source, file_type, chunks, ingested_at."""
    client = get_client()
    if not client.collection_exists(config.COLLECTION_NAME):
        return []
    return list_sources(client, config.COLLECTION_NAME)


def remove_document(source: str, *, delete_file: bool = False) -> str:
    """Remove one document's chunks from the index, prune the manifest,
    and optionally delete the uploaded file from disk."""
    if not source or Path(source).name != source:
        raise ValueError(f"Invalid source name: {source!r}")

    messages: list[str] = []

    client = get_client()
    if client.collection_exists(config.COLLECTION_NAME):
        delete_points_by_source(client, config.COLLECTION_NAME, source)
        messages.append(f"Deleted chunks for `{source}` from `{config.COLLECTION_NAME}`.")
    else:
        messages.append(f"Collection `{config.COLLECTION_NAME}` does not exist.")

    manifest = load_manifest()
    stale_keys = [k for k in manifest if Path(k).name == source]
    for key in stale_keys:
        del manifest[key]
    if stale_keys:
        save_manifest(manifest)
        messages.append("Removed manifest entry (file will be re-indexed if ingested again).")

    if delete_file:
        target = (UPLOAD_DIR / source).resolve()
        upload_root = UPLOAD_DIR.resolve()
        if target.parent == upload_root and target.is_file():
            target.unlink()
            messages.append(f"Deleted uploaded file `{source}`.")
        else:
            messages.append(f"Uploaded file `{source}` not found in {upload_root} (left on disk).")

    return " ".join(messages)


def create_index() -> str:
    """Create the collection if missing (loads the embedding model)."""
    store = build_store()
    status = store.status()
    return (
        f"Collection `{config.COLLECTION_NAME}` is ready "
        f"({status['points']} point(s))."
    )


def reset_index() -> str:
    """Drop the whole collection, recreate it empty, and clear the manifest."""
    store = build_store()
    store.reset()
    save_manifest({})
    # Clear model config so next indexing can choose freely
    if MODEL_CONFIG_PATH.exists():
        MODEL_CONFIG_PATH.unlink()
    return f"Index `{config.COLLECTION_NAME}` reset: all chunks deleted, manifest cleared."


def prepare_document(
    path: Path,
    chunker,
    *,
    extra_metadata: dict[str, Any] | None = None,
):
    """Parse, attach metadata, and chunk a single file."""
    document = file_to_document(path)
    apply_document_metadata(document, build_file_metadata(path, extra=extra_metadata))
    chunker.chunk_document(document)
    annotate_chunks(document)
    return document


def ingest_paths(paths, *, chunker_config=DEFAULT_CHUNKER_CONFIG_PATH, store=None,
                  extra_metadata=None, force: bool = False, use_quantized: bool = True) -> int:
    ensure_chunker_config(chunker_config)
    chunker = load_chunker(chunker_config)
    vector_store = store or build_store(use_quantized=use_quantized)
    manifest = load_manifest()

    total_chunks = 0
    for raw in paths:
        path = Path(raw).resolve()
        if not path.is_file():
            logger.warning("Skipping missing file: %s", path)
            continue

        key = str(path)
        cached = manifest.get(key)

        if not force and is_unchanged(path, cached):
            logger.info("Skipping unchanged file: %s", path)
            continue

        if cached is not None:
            logger.info("File changed, removing old chunks: %s", path)
            vector_store.delete_by_source(path.name)

        document = prepare_document(path, chunker, extra_metadata=extra_metadata)
        if not document.chunks:
            logger.warning("No chunks produced for %s", path)
            continue

        vector_store.write(document.chunks)
        total_chunks += len(document.chunks)

        manifest[key] = {**file_fingerprint(path), "hash": file_hash(path), "chunk_count": len(document.chunks)}
        logger.info("Indexed %d chunk(s) from %s", len(document.chunks), path.name)

    save_manifest(manifest)
    save_model_config(use_quantized)
    return total_chunks

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument(
        "files",
        nargs="+",
        help="Paths to documents (pdf, docx, txt, md, csv, xlsx, …)",
    )
    parser.add_argument(
        "--chunker-config",
        default=DEFAULT_CHUNKER_CONFIG_PATH,
        help="JSON config for RecursiveChunker (created with defaults if missing)",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra document metadata (repeatable), e.g. --meta category=research",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags stored on every chunk from this run",
    )
    args = parser.parse_args()

    extra: dict[str, Any] = parse_extra_metadata(args.meta)
    if args.tags.strip():
        extra["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]

    count = ingest_paths(
        args.files,
        chunker_config=args.chunker_config,
        extra_metadata=extra or None,
    )
    print(f"Done. Indexed {count} chunk(s) from {len(args.files)} file(s).")


if __name__ == "__main__":
    main()
