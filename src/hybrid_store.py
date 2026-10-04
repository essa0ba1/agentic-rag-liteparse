"""Qdrant store with dense + sparse (BM25) vectors and RRF hybrid search."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional, Union

from chonkie.embeddings import BaseEmbeddings
from chonkie.types import Chunk
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion

import config

logger = logging.getLogger(__name__)

_sparse_model: Any | None = None


def _get_sparse_model(model_name: str):
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding

        logger.info("Loading sparse embedding model: %s", model_name)
        _sparse_model = SparseTextEmbedding(model_name=model_name)
    return _sparse_model


def _merge_chunk_payload(chunk: Chunk, fields: dict[str, Any]) -> dict[str, Any]:
    raw = getattr(chunk, "metadata", None)
    extra: dict[str, Any] = raw if isinstance(raw, dict) else {}
    merged = {**extra, **fields}
    out: dict[str, Any] = {}
    for key, val in merged.items():
        if isinstance(val, (str, int, float, bool)):
            out[key] = val
        elif val is None:
            continue
        else:
            out[key] = json.dumps(val, default=str)
    return out


class HybridQdrantStore:
    """Write chunks with dense + sparse vectors; search with reciprocal rank fusion."""

    def __init__(
        self,
        embedding_model: BaseEmbeddings,
        *,
        url: str = config.QDRANT_URL,
        collection_name: str = config.COLLECTION_NAME,
        api_key: str | None = None,
        sparse_model_name: str = config.SPARSE_MODEL_NAME,
        dense_vector_name: str = config.DENSE_VECTOR_NAME,
        sparse_vector_name: str = config.SPARSE_VECTOR_NAME,
    ) -> None:
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.sparse_model_name = sparse_model_name
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.dimension = embedding_model.dimension

        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(url=url)

        self._ensure_collection()
        self._ensure_payload_indexes()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            params = info.config.params
            has_dense = bool(params.vectors)
            has_sparse = bool(params.sparse_vectors)
            if not has_dense or not has_sparse:
                raise RuntimeError(
                    f"Collection {self.collection_name!r} is not hybrid-ready "
                    f"(needs named dense + sparse vectors). Delete it and re-run ingestion."
                )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.dense_vector_name: models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info("Created hybrid Qdrant collection %r", self.collection_name)

    def _ensure_payload_indexes(self) -> None:
        for field_name, schema in (
            ("source", "keyword"),
            ("file_type", "keyword"),
            ("text", "text"),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:
                logger.debug("Payload index %s skipped: %s", field_name, exc)

    @staticmethod
    def _point_id(chunk: Chunk) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{chunk.id}:{chunk.text[:256]}"))

    def _payload(self, chunk: Chunk) -> dict[str, Any]:
        return _merge_chunk_payload(
            chunk,
            {
                "text": chunk.text,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                "token_count": chunk.token_count,
            },
        )

    def _sparse_embed_passages(self, texts: list[str]) -> list[models.SparseVector]:
        sparse_model = _get_sparse_model(self.sparse_model_name)
        return [
            models.SparseVector(
                indices=vec.indices.tolist(),
                values=vec.values.tolist(),
            )
            for vec in sparse_model.embed(texts)
        ]

    def _sparse_embed_query(self, query: str) -> models.SparseVector:
        sparse_model = _get_sparse_model(self.sparse_model_name)
        vec = list(sparse_model.query_embed(query=query))[0]
        return models.SparseVector(
            indices=vec.indices.tolist(),
            values=vec.values.tolist(),
        )

    def write(self, chunks: Union[Chunk, list[Chunk]]) -> None:
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        if not chunks:
            return

        texts = [c.text for c in chunks]
        dense_vectors = [self.embedding_model.embed(t).tolist() for t in texts]
        sparse_vectors = self._sparse_embed_passages(texts)

        points = [
            models.PointStruct(
                id=self._point_id(chunk),
                vector={
                    self.dense_vector_name: dense,
                    self.sparse_vector_name: sparse,
                },
                payload=self._payload(chunk),
            )
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        logger.info("Indexed %d chunk(s) in %r", len(points), self.collection_name)

    def search(
        self,
        query: str,
        *,
        limit: int = config.TOP_K,
        query_filter: models.Filter | None = None,
        prefetch_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        prefetch_limit = prefetch_limit or max(limit * config.HYBRID_PREFETCH_MULTIPLIER, limit)

        dense_vector = self.embedding_model.embed(query).tolist()
        sparse_vector = self._sparse_embed_query(query)

        if config.USE_HYBRID_SEARCH:
            dense_request = models.QueryRequest(
                query=dense_vector,
                using=self.dense_vector_name,
                filter=query_filter,
                limit=prefetch_limit,
                with_payload=True,
            )
            sparse_request = models.QueryRequest(
                query=sparse_vector,
                using=self.sparse_vector_name,
                filter=query_filter,
                limit=prefetch_limit,
                with_payload=True,
            )
            dense_response, sparse_response = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=[dense_request, sparse_request],
            )
            fused = reciprocal_rank_fusion(
                [dense_response.points, sparse_response.points],
                limit=limit,
            )
            points = fused
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using=self.dense_vector_name,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            points = response.points

        return [
            {"id": point.id, "score": point.score, **(point.payload or {})}
            for point in points
        ]
    def delete_by_source(self, source: str) -> None:
        """Delete every chunk whose payload `source` matches (the file name)."""
        delete_points_by_source(self.client, self.collection_name, source)

    def list_sources(self) -> list[dict[str, Any]]:
        """Aggregate indexed chunks per source document (payload-only scan)."""
        return list_sources(self.client, self.collection_name)

    def status(self) -> dict[str, Any]:
        """Collection existence + point count."""
        return collection_status(self.client, self.collection_name)

    def reset(self) -> None:
        """Drop the collection and recreate it empty (keeps vector config)."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info("Dropped collection %r", self.collection_name)
        self._ensure_collection()
        self._ensure_payload_indexes()


# ── client-only management helpers (no embedding model required) ─────────────


def delete_points_by_source(client: QdrantClient, collection_name: str, source: str) -> None:
    """Delete every chunk whose payload `source` matches (the file name)."""
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
            )
        ),
        wait=True,
    )
    logger.info("Deleted chunks for source %r in %r", source, collection_name)


def list_sources(client: QdrantClient, collection_name: str) -> list[dict[str, Any]]:
    """Aggregate indexed chunks per source document (payload-only scan)."""
    stats: dict[str, dict[str, Any]] = {}
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for rec in records:
            payload = rec.payload or {}
            name = payload.get("source", "unknown")
            entry = stats.setdefault(
                name,
                {
                    "source": name,
                    "file_type": payload.get("file_type", ""),
                    "chunks": 0,
                    "ingested_at": "",
                },
            )
            entry["chunks"] += 1
            ingested_at = payload.get("ingested_at") or ""
            if ingested_at > entry["ingested_at"]:
                entry["ingested_at"] = ingested_at
        if offset is None:
            break
    return sorted(stats.values(), key=lambda e: e["source"])


def collection_status(client: QdrantClient, collection_name: str) -> dict[str, Any]:
    """Collection existence + point count."""
    if not client.collection_exists(collection_name):
        return {"exists": False, "points": 0}
    info = client.get_collection(collection_name)
    return {"exists": True, "points": info.points_count or 0}




