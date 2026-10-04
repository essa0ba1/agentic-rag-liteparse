"""Metadata helpers for ingest and retrieval."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_retrieval_ui_sink: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "retrieval_ui_sink",
    default=None,
)

from chonkie.types import Chunk, Document


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_file_metadata(
    file_path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Base document-level metadata derived from the file on disk."""
    path = Path(file_path).resolve()
    stat = path.stat()
    meta: dict[str, Any] = {
        "source": path.name,
        "path": str(path),
        "file_type": path.suffix.lower().lstrip(".") or "unknown",
        "file_size_bytes": stat.st_size,
        "ingested_at": _utc_now_iso(),
    }
    if extra:
        meta.update(extra)
    return meta


def parse_extra_metadata(pairs: list[str]) -> dict[str, str]:
    """Parse repeated ``key=value`` CLI flags into a flat metadata dict."""
    out: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid metadata {item!r}; use key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid metadata {item!r}; empty key")
        out[key] = value.strip()
    return out


def apply_document_metadata(document: Document, metadata: dict[str, Any]) -> Document:
    document.metadata = {**document.metadata, **metadata}
    return document


def annotate_chunks(document: Document) -> list[Chunk]:
    """Add per-chunk indices and merge document metadata onto each chunk."""
    total = len(document.chunks)
    for index, chunk in enumerate(document.chunks):
        chunk.metadata = {
            **document.metadata,
            **chunk.metadata,
            "chunk_index": index,
            "total_chunks": total,
        }
    return document.chunks


def format_hit_for_llm(hit: dict[str, Any], rank: int) -> str:
    """Render one search hit for the agent tool output."""
    source = hit.get("source") or (hit.get("metadata") or {}).get("source", "unknown")
    file_type = hit.get("file_type", "")
    chunk_index = hit.get("chunk_index")
    score = hit.get("score")
    parts = [f"[{rank}] (source: {source}"]
    if file_type:
        parts.append(f"type: {file_type}")
    if chunk_index is not None:
        parts.append(f"chunk: {chunk_index}")
    if score is not None:
        parts.append(f"score: {score:.4f}")
    parts.append(")")
    header = ", ".join(parts)

    extra_keys = (
        "title",
        "category",
        "tags",
        "path",
        "ingested_at",
    )
    extras = {k: hit[k] for k in extra_keys if k in hit and hit[k] not in (None, "")}
    if extras:
        header += f"\nmetadata: {json.dumps(extras, ensure_ascii=False)}"

    text = hit.get("text", "")
    return f"{header}\n{text}"


def begin_retrieval_capture() -> tuple[list[dict[str, Any]], Token]:
    """Collect retrieval events until `end_retrieval_capture` (for UI / tests)."""
    events: list[dict[str, Any]] = []
    token = _retrieval_ui_sink.set(events)
    return events, token


def end_retrieval_capture(token: Token) -> None:
    _retrieval_ui_sink.reset(token)


def hit_summary(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    """Structured fields for logging one retrieval hit."""
    meta = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    source = hit.get("source") or meta.get("source")
    return {
        "rank": rank,
        "id": hit.get("id"),
        "score": hit.get("score"),
        "source": source,
        "file_type": hit.get("file_type") or meta.get("file_type"),
        "chunk_index": hit.get("chunk_index", meta.get("chunk_index")),
        "text": hit.get("text", ""),
    }


def _push_retrieval_ui_event(query: str, summaries: list[dict[str, Any]]) -> None:
    sink = _retrieval_ui_sink.get()
    if sink is not None:
        sink.append({"query": query, "hits": summaries})


def format_retrieval_events_for_ui(
    events: list[dict[str, Any]],
    *,
    user_question: str | None = None,
    text_chars: int = 800,
) -> str:
    """Markdown block for Gradio retrieval panel."""
    if not events:
        if user_question:
            return f"**Question:** {user_question}\n\n*No `search_qdrant` calls this turn.*"
        return "*No retrievals yet.*"

    sections: list[str] = []
    if user_question:
        sections.append(f"**Question:** {user_question}")
    for idx, event in enumerate(events, start=1):
        q = event.get("query", "")
        hits: list[dict[str, Any]] = event.get("hits") or []
        sections.append(f"#### Retrieval {idx}\n**Query:** `{q}` · **{len(hits)}** chunk(s)")
        if not hits:
            sections.append("*No chunks returned.*")
            continue
        for row in hits:
            score = row.get("score")
            score_s = f"{score:.4f}" if score is not None else "n/a"
            text = (row.get("text") or "").strip()
            if len(text) > text_chars:
                text = text[:text_chars] + "…"
            sections.append(
                f"**[{row.get('rank')}]** `{row.get('source')}` · chunk `{row.get('chunk_index')}` "
                f"· score `{score_s}`\n\n{text}"
            )
    return "\n\n".join(sections)


def log_retrieved_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    log_path: str | Path | None = None,
    preview_chars: int = 400,
    logger: logging.Logger | None = None,
) -> None:
    """Log retrieved chunks to the application logger and optionally append JSONL."""
    log = logger or logging.getLogger(__name__)
    summaries = [hit_summary(h, i) for i, h in enumerate(hits, start=1)]
    _push_retrieval_ui_event(query, summaries)

    if not hits:
        log.info("Retrieval query=%r → 0 hits", query)
        return
    log.info("Retrieval query=%r → %d hit(s)", query, len(summaries))
    for row in summaries:
        text = row["text"] or ""
        preview = text[:preview_chars] + ("…" if len(text) > preview_chars else "")
        log.info(
            "  [%s] source=%s chunk=%s score=%s id=%s\n%s",
            row["rank"],
            row["source"],
            row["chunk_index"],
            f"{row['score']:.4f}" if row["score"] is not None else "n/a",
            row["id"],
            preview,
        )

    if not log_path:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "retrieved_at": _utc_now_iso(),
        "query": query,
        "hits": summaries,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
