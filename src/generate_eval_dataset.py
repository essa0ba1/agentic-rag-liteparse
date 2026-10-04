#!/usr/bin/env python3
"""
Generate a synthetic retrieval-evaluation dataset from chunks that are
ALREADY stored in a Qdrant collection.

For a stratified sample of chunks (by doc_type and position-in-document),
ask an LLM for:

    - positive_query:
        A question that this exact chunk directly and fully answers.

    - hard_negative_query:
        A plausible question that sounds like it belongs to the same
        document/domain, but is NOT answered by this chunk.

Chunks are read with Qdrant `scroll` (no re-parsing, no re-chunking), so the
eval chunks are exactly the production chunks, and `chunk_id` is the Qdrant
point id, so it can be compared directly with retrieval results.

Example:

    # .env (loaded automatically)
    CEREBRAS_API_KEY=csk-...
    QDRANT_URL=http://localhost:6333   # optional, this is the default

    python generate_eval_dataset.py \
        --collection my_collection \
        --sample-frac 0.2 \
        --out eval_dataset.jsonl

Payload field names are configurable (dotted paths allowed, e.g.
"metadata.doc_type"):

    --text-field text --source-field source --doc-type-field doc_type \
    --start-field start_index --end-field end_index
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# Load variables from a .env file (searched upward from the current dir).
load_dotenv()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_API_KEY_ENV = "CEREBRAS_API_KEY"

# Local Docker Qdrant, no API key. Override with QDRANT_URL in .env if needed.
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

DEFAULT_MODEL = "qwen-3.8-27b"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.95
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_REASONING_FORMAT = "parsed"
DEFAULT_MAX_COMPLETION_TOKENS = 2048

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

GEN_PROMPT = """
You are generating a high-quality retrieval evaluation dataset.

You will receive ONE excerpt from a "{doc_type}" document.

Generate exactly TWO questions.

1. positive_query

Write a realistic user question that this excerpt directly and fully
answers.

Requirements:
- The answer must be explicitly supported by this excerpt.
- Do not require information from other parts of the document.
- Phrase the question naturally, as a real user would ask it.
- Do not copy sentences or phrases verbatim from the excerpt.
- Avoid questions that are too broad.
- Prefer specific questions whose answer is clearly contained in the chunk.

2. hard_negative_query

Write a realistic question that:

- sounds like it belongs to this same document,
- uses similar terminology/entities,
- is about the same general subject,
- but is NOT answered by this excerpt,
- and would require another clause, paragraph, table, or section
  to answer correctly.

Important:
- The negative must be a believable near-miss.
- Do NOT make it obviously unrelated.
- Do NOT create a question whose answer can be inferred directly from
  the supplied excerpt.
- Avoid simply changing one irrelevant word.
- The negative should represent a realistic retrieval failure.

Return ONLY valid JSON with exactly these two keys:

{{
  "positive_query": "...",
  "hard_negative_query": "..."
}}

Excerpt:

"{chunk_text}"
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SampledChunk:
    chunk_id: str
    source: str
    doc_type: str
    bucket: int
    chunk_text: str


@dataclass
class EvalRow:
    chunk_id: str
    source: str
    doc_type: str
    bucket: int
    chunk_text: str
    positive_query: str
    hard_negative_query: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_field(payload: dict[str, Any] | None, path: str) -> Any:
    """Read a (possibly dotted) field from a payload, e.g. 'metadata.doc_type'."""
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def position_bucket(
    position: float,
    total: float,
    n_buckets: int = 3,
) -> int:
    """Early / middle / late bucket (0, 1, 2 for n_buckets=3)."""
    if total <= 0:
        return 0
    frac = min(max(position / total, 0.0), 0.999999)
    return int(frac * n_buckets)


# ---------------------------------------------------------------------------
# Qdrant chunk collection
# ---------------------------------------------------------------------------

def make_qdrant_client(args: argparse.Namespace) -> QdrantClient:
    if args.qdrant_path:
        return QdrantClient(path=args.qdrant_path)

    return QdrantClient(
        url=args.qdrant_url,
        timeout=args.qdrant_timeout,
    )


def collect_chunks_from_qdrant(
    client: QdrantClient,
    args: argparse.Namespace,
) -> list[SampledChunk]:
    """
    Scroll the whole collection (payload only, no vectors) and build
    SampledChunk objects with a position bucket per chunk.

    Position within a document is computed per `source`:
      - preferred: start_index / max(end_index) of that source
      - fallback : ordinal rank of the chunk within its source
    """

    scroll_filter = None
    if args.filter_doc_type:
        scroll_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key=args.doc_type_field,
                    match=qm.MatchAny(any=args.filter_doc_type),
                )
            ]
        )

    raw: list[dict[str, Any]] = []
    offset = None
    skipped_no_text = 0

    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            scroll_filter=scroll_filter,
            limit=args.scroll_batch,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for p in points:
            payload = p.payload or {}

            text = get_field(payload, args.text_field)
            if not isinstance(text, str) or not text.strip():
                skipped_no_text += 1
                continue

            source = get_field(payload, args.source_field)
            source = str(source) if source is not None else "unknown"

            doc_type = get_field(payload, args.doc_type_field)
            if not doc_type:
                # Fallback: parent folder name of the source path.
                doc_type = Path(source).parent.name or "unknown"

            raw.append(
                {
                    "id": str(p.id),
                    "text": text,
                    "source": source,
                    "doc_type": str(doc_type),
                    "start": to_int(get_field(payload, args.start_field)),
                    "end": to_int(get_field(payload, args.end_field)),
                }
            )

        if offset is None:
            break

    logger.info(
        "Scrolled %d usable chunks from collection '%s' "
        "(skipped %d without text)",
        len(raw),
        args.collection,
        skipped_no_text,
    )

    # Group by source to compute per-document positions.
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in raw:
        by_source[r["source"]].append(r)

    out: list[SampledChunk] = []
    ordinal_fallback_docs = 0

    for source, rows in by_source.items():
        have_offsets = all(r["start"] is not None for r in rows)

        if have_offsets:
            doc_length = max(
                (r["end"] if r["end"] is not None else r["start"])
                for r in rows
            )
            doc_length = max(doc_length, 1)
            for r in rows:
                r["bucket"] = position_bucket(r["start"], doc_length)
        else:
            # No offsets in payload: rank by scroll order within the doc.
            ordinal_fallback_docs += 1
            n = len(rows)
            for i, r in enumerate(rows):
                r["bucket"] = position_bucket(i, n)

    if ordinal_fallback_docs:
        logger.warning(
            "%d documents had no '%s' in payload; position buckets fell back "
            "to scroll order, which may not match true document order.",
            ordinal_fallback_docs,
            args.start_field,
        )

    for rows in by_source.values():
        for r in rows:
            out.append(
                SampledChunk(
                    chunk_id=r["id"],
                    source=r["source"],
                    doc_type=r["doc_type"],
                    bucket=r["bucket"],
                    chunk_text=r["text"],
                )
            )

    return out


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    chunks: list[SampledChunk],
    sample_frac: float,
    seed: int,
    min_per_group: int = 1,
) -> list[SampledChunk]:
    """Sample independently inside each (doc_type, position_bucket) group."""

    if not 0 < sample_frac <= 1:
        raise ValueError("--sample-frac must be > 0 and <= 1")

    rng = random.Random(seed)

    groups: dict[tuple[str, int], list[SampledChunk]] = defaultdict(list)
    for chunk in chunks:
        groups[(chunk.doc_type, chunk.bucket)].append(chunk)

    sampled: list[SampledChunk] = []

    for key, group in sorted(groups.items()):
        n = max(min_per_group, round(len(group) * sample_frac))
        n = min(n, len(group))

        sampled.extend(rng.sample(group, n))

        logger.info(
            "doc_type=%s bucket=%s: sampled %d/%d",
            key[0],
            key[1],
            n,
            len(group),
        )

    return sampled


# ---------------------------------------------------------------------------
# JSON cleaning
# ---------------------------------------------------------------------------

def clean_json_response(raw: str) -> str:
    """Strip markdown fences and stray <think>...</think> blocks."""

    raw = raw.strip()

    if "<think>" in raw:
        raw = THINK_BLOCK_RE.sub("", raw).strip()

    # Unterminated <think> (truncated mid-reasoning): nothing to recover.
    if raw.startswith("<think>"):
        return ""

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    return raw


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm_for_queries(
    client: Cerebras,
    model: str,
    doc_type: str,
    chunk_text: str,
    max_retries: int = 3,
    max_chunk_chars: int = 6000,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    reasoning_format: str = DEFAULT_REASONING_FORMAT,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> dict[str, str] | None:

    prompt = GEN_PROMPT.format(
        doc_type=doc_type,
        chunk_text=chunk_text[:max_chunk_chars],
    )

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate retrieval evaluation data. "
                    "Follow the requested JSON format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_effort": reasoning_effort,
        "reasoning_format": reasoning_format,
    }

    for attempt in range(1, max_retries + 1):

        try:
            response = client.chat.completions.create(**request_kwargs)

            if not response.choices:
                logger.warning(
                    "Attempt %d/%d: provider returned no choices",
                    attempt,
                    max_retries,
                )
                time.sleep(1.5 * attempt)
                continue

            content = response.choices[0].message.content

            if not content:
                logger.warning(
                    "Attempt %d/%d: empty content (finish_reason=%s)",
                    attempt,
                    max_retries,
                    response.choices[0].finish_reason,
                )
                time.sleep(1.5 * attempt)
                continue

            raw = clean_json_response(content)

            if not raw:
                logger.warning(
                    "Attempt %d/%d: only reasoning/whitespace after cleanup "
                    "(finish_reason=%s, raw len=%d). Likely truncated by "
                    "max_completion_tokens.",
                    attempt,
                    max_retries,
                    response.choices[0].finish_reason,
                    len(content),
                )
                time.sleep(1.5 * attempt)
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Attempt %d/%d: invalid JSON: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                logger.debug("Raw response: %s", raw[:500])
                time.sleep(1.5 * attempt)
                continue

            positive = data.get("positive_query")
            negative = data.get("hard_negative_query")

            if not isinstance(positive, str) or not isinstance(negative, str):
                logger.warning("Missing/invalid query fields: %r", data)
                time.sleep(1.5 * attempt)
                continue

            positive = positive.strip()
            negative = negative.strip()

            if len(positive) < 5 or len(negative) < 5:
                logger.warning(
                    "Query too short: %r / %r",
                    positive,
                    negative,
                )
                time.sleep(1.0)
                continue

            if positive.lower() == negative.lower():
                logger.warning("Positive and negative queries are identical")
                time.sleep(1.0)
                continue

            return {
                "positive_query": positive,
                "hard_negative_query": negative,
            }

        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed: %s",
                attempt,
                max_retries,
                exc,
            )
            time.sleep(1.5 * attempt)

    return None


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, row: EvalRow) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---------------- Qdrant source ----------------

    parser.add_argument(
        "--collection",
        required=True,
        help="Name of the existing Qdrant collection to sample chunks from.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Qdrant server URL (default: $QDRANT_URL or "
             "http://localhost:6333). Ignored if --qdrant-path is given.",
    )
    parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Path to a local on-disk Qdrant database (instead of a server).",
    )
    parser.add_argument("--qdrant-timeout", type=int, default=60)
    parser.add_argument(
        "--scroll-batch",
        type=int,
        default=256,
        help="Points fetched per scroll request.",
    )

    # ---------------- Payload schema ----------------

    parser.add_argument("--text-field", default="text",
                        help="Payload field holding the chunk text.")
    parser.add_argument("--source-field", default="source",
                        help="Payload field identifying the parent document.")
    parser.add_argument("--doc-type-field", default="doc_type",
                        help="Payload field holding the doc_type. If missing "
                             "on a point, the source's parent folder name is used.")
    parser.add_argument("--start-field", default="start_index",
                        help="Payload field with the chunk's start offset.")
    parser.add_argument("--end-field", default="end_index",
                        help="Payload field with the chunk's end offset.")
    parser.add_argument(
        "--filter-doc-type",
        action="append",
        default=None,
        help="Only sample chunks with this doc_type. Can be repeated.",
    )

    # ---------------- Sampling ----------------

    parser.add_argument("--sample-frac", type=float, default=0.20,
                        help="Fraction sampled from each "
                             "(doc_type, position_bucket) group.")
    parser.add_argument("--min-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # ---------------- Cerebras ----------------

    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV,
                        help="Env var containing the Cerebras API key.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        choices=["low", "medium", "high"])
    parser.add_argument("--reasoning-format", default=DEFAULT_REASONING_FORMAT,
                        choices=["none", "parsed", "raw", "hidden"])

    # ---------------- Generation ----------------

    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-chunk-chars", type=int, default=6000)
    parser.add_argument("--max-completion-tokens", type=int,
                        default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--temperature", type=float,
                        default=DEFAULT_TEMPERATURE)

    # ---------------- Output ----------------

    parser.add_argument("--out", default="eval_dataset.jsonl")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = build_parser()
    args = parser.parse_args()

    if not 0 < args.sample_frac <= 1:
        parser.error("--sample-frac must be > 0 and <= 1")

    if args.min_per_group < 1:
        parser.error("--min-per-group must be >= 1")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"Missing API key.\nSet {args.api_key_env} first.\n")

    logger.info("Provider: Cerebras | Model: %s", args.model)
    llm = Cerebras(api_key=api_key)

    # ---------------- Load chunks from Qdrant ----------------

    qdrant = make_qdrant_client(args)

    if not qdrant.collection_exists(args.collection):
        sys.exit(f"Collection '{args.collection}' does not exist.")

    all_chunks = collect_chunks_from_qdrant(qdrant, args)

    if not all_chunks:
        sys.exit(
            "No chunks collected. Check --collection and the payload "
            "field names (--text-field, --source-field, ...)."
        )

    logger.info("Total chunks collected: %d", len(all_chunks))

    # ---------------- Sample ----------------

    sampled = stratified_sample(
        chunks=all_chunks,
        sample_frac=args.sample_frac,
        seed=args.seed,
        min_per_group=args.min_per_group,
    )

    logger.info("Total sampled chunks: %d", len(sampled))

    # ---------------- Output file ----------------

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    # ---------------- Generate ----------------

    successful = 0
    failed = 0
    total = len(sampled)

    for i, chunk in enumerate(sampled, start=1):

        logger.info("[%d/%d] Generating queries for %s", i, total, chunk.chunk_id)

        result = call_llm_for_queries(
            client=llm,
            model=args.model,
            doc_type=chunk.doc_type,
            chunk_text=chunk.chunk_text,
            max_retries=args.max_retries,
            max_chunk_chars=args.max_chunk_chars,
            temperature=args.temperature,
            top_p=args.top_p,
            reasoning_effort=args.reasoning_effort,
            reasoning_format=args.reasoning_format,
            max_completion_tokens=args.max_completion_tokens,
        )

        if result is None:
            failed += 1
            logger.warning("Skipping chunk %s after retries", chunk.chunk_id)
            continue

        append_jsonl(
            out_path,
            EvalRow(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                doc_type=chunk.doc_type,
                bucket=chunk.bucket,
                chunk_text=chunk.chunk_text,
                positive_query=result["positive_query"],
                hard_negative_query=result["hard_negative_query"],
            ),
        )

        successful += 1

        logger.info("  positive: %s", result["positive_query"])
        logger.info("  negative: %s", result["hard_negative_query"])

    # ---------------- Summary ----------------

    logger.info("=" * 70)
    logger.info("Generation complete")
    logger.info("Sampled:    %d", total)
    logger.info("Successful: %d", successful)
    logger.info("Failed:     %d", failed)
    logger.info("Output:     %s", out_path)
    logger.info("=" * 70)

    print(
        f"\nDone.\n"
        f"  Sampled:    {total}\n"
        f"  Successful: {successful}\n"
        f"  Failed:     {failed}\n"
        f"  Output:     {out_path}\n\n"
        f"Before using this dataset for metrics, manually inspect "
        f"a representative subset.\n"
        f"Pay particular attention to:\n"
        f"  1. positive queries that are not fully answerable by the chunk\n"
        f"  2. hard negatives that are accidentally answerable by the chunk\n"
    )


if __name__ == "__main__":
    main()