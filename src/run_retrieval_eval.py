"""Evaluate retrieval quality with and without BM25 (sparse) fusion.

Loads the JSONL dataset produced by generate_eval_dataset.py, runs every
positive_query and hard_negative_query through the real Qdrant collection in
two modes -- dense-only, and dense+BM25 hybrid (RRF) -- and reports:

  Per query type:
    - recall@k / precision@k on positive queries (did the source chunk come back?)
    - MRR on positive queries (how high did it rank?)
    - leak rate@k on hard-negative queries (did the source chunk come back for
      a query it should NOT have matched? lower is better)

  Overall (positives + hard negatives together), per mode:
    Every (query, source chunk) pair is a yes/no decision. The system says
    "yes" when the source chunk is in the top-k.
        positive query      + chunk in top-k  -> TP
        positive query      + chunk missing   -> FN
        hard-negative query + chunk in top-k  -> FP  (leak)
        hard-negative query + chunk missing   -> TN
    From these: precision, recall, F1, accuracy.

Usage:
    python run_retrieval_eval.py --dataset eval_dataset.jsonl --k 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from qdrant_client.http import models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion

import config
from data_pipeline import build_store
from hybrid_store import HybridQdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RowResult:
    chunk_id: str
    doc_type: str
    query_type: str  # "positive" or "hard_negative"
    mode: str  # "dense" or "hybrid"
    rank: int | None  # 1-indexed position of chunk_id in results, or None if absent
    k: int


def dense_search_ids(store: HybridQdrantStore, query: str, limit: int) -> list[str]:
    dense_vector = store.embedding_model.embed(query).tolist()
    response = store.client.query_points(
        collection_name=store.collection_name,
        query=dense_vector,
        using=store.dense_vector_name,
        limit=limit,
        with_payload=False,
    )
    return [str(p.id) for p in response.points]


def hybrid_search_ids(store: HybridQdrantStore, query: str, limit: int, prefetch_mult: int = 4) -> list[str]:
    prefetch_limit = max(limit * prefetch_mult, limit)
    dense_vector = store.embedding_model.embed(query).tolist()
    sparse_vector = store._sparse_embed_query(query)

    dense_request = models.QueryRequest(
        query=dense_vector, using=store.dense_vector_name, limit=prefetch_limit, with_payload=False
    )
    sparse_request = models.QueryRequest(
        query=sparse_vector, using=store.sparse_vector_name, limit=prefetch_limit, with_payload=False
    )
    dense_resp, sparse_resp = store.client.query_batch_points(
        collection_name=store.collection_name,
        requests=[dense_request, sparse_request],
    )
    fused = reciprocal_rank_fusion([dense_resp.points, sparse_resp.points], limit=limit)
    return [str(p.id) for p in fused]


def rank_of(target_id: str, result_ids: list[str]) -> int | None:
    try:
        return result_ids.index(target_id) + 1  # 1-indexed
    except ValueError:
        return None


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def summarize(results: list[RowResult], k: int) -> dict[str, dict[str, float]]:
    """Aggregate per (mode, query_type)."""
    summary: dict[tuple[str, str], list[RowResult]] = {}
    for r in results:
        summary.setdefault((r.mode, r.query_type), []).append(r)

    out: dict[str, dict[str, float]] = {}
    for (mode, qtype), rows in summary.items():
        n = len(rows)
        hits = [1 if r.rank is not None else 0 for r in rows]
        recall_at_k = safe_div(sum(hits), n)
        precision_at_k = safe_div(sum(h / k for h in hits), n)
        reciprocal_ranks = [1.0 / r.rank if r.rank else 0.0 for r in rows]
        mrr = safe_div(sum(reciprocal_ranks), n)

        out[f"{mode}_{qtype}"] = {
            "n": n,
            "recall_at_k": round(recall_at_k, 4),
            "precision_at_k": round(precision_at_k, 4),
            "mrr": round(mrr, 4),
        }
    return out


def summarize_overall(results: list[RowResult]) -> dict[str, dict[str, float]]:
    """Overall precision/recall/F1 per mode, treating each (query, chunk) pair
    as a yes/no decision (see module docstring for the confusion matrix)."""
    out: dict[str, dict[str, float]] = {}

    for mode in sorted({r.mode for r in results}):
        rows = [r for r in results if r.mode == mode]

        tp = sum(1 for r in rows if r.query_type == "positive" and r.rank is not None)
        fn = sum(1 for r in rows if r.query_type == "positive" and r.rank is None)
        fp = sum(1 for r in rows if r.query_type == "hard_negative" and r.rank is not None)
        tn = sum(1 for r in rows if r.query_type == "hard_negative" and r.rank is None)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        accuracy = safe_div(tp + tn, tp + fn + fp + tn)

        out[mode] = {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }
    return out


def print_report(summary: dict[str, dict[str, float]], k: int) -> None:
    print(f"\n=== Retrieval eval (k={k}) ===\n")
    header = f"{'mode':<10}{'query_type':<15}{'n':>5}{'recall@k':>12}{'precision@k':>14}{'mrr':>10}"
    print(header)
    print("-" * len(header))
    for mode in ("dense", "hybrid"):
        for qtype in ("positive", "hard_negative"):
            row = summary.get(f"{mode}_{qtype}")
            if not row:
                continue
            print(
                f"{mode:<10}{qtype:<15}{row['n']:>5}{row['recall_at_k']:>12.4f}"
                f"{row['precision_at_k']:>14.4f}{row['mrr']:>10.4f}"
            )
    print(
        "\nNote: for query_type=hard_negative, 'recall@k' here means the source chunk\n"
        "STILL showed up for a query it shouldn't answer -- i.e. this is a leak rate,\n"
        "lower is better (0.0 = perfect discrimination). precision@k/mrr are not\n"
        "meaningful for hard_negative rows and can be ignored for that line.\n"
    )


def print_overall_report(overall: dict[str, dict[str, float]], k: int) -> None:
    print(f"=== Overall (positives + hard negatives, k={k}) ===\n")
    header = (
        f"{'mode':<10}{'TP':>5}{'FN':>5}{'FP':>5}{'TN':>5}"
        f"{'precision':>12}{'recall':>10}{'f1':>10}{'accuracy':>11}"
    )
    print(header)
    print("-" * len(header))
    for mode in ("dense", "hybrid"):
        row = overall.get(mode)
        if not row:
            continue
        print(
            f"{mode:<10}{row['tp']:>5}{row['fn']:>5}{row['fp']:>5}{row['tn']:>5}"
            f"{row['precision']:>12.4f}{row['recall']:>10.4f}{row['f1']:>10.4f}{row['accuracy']:>11.4f}"
        )
    print(
        "\nA 'yes' means the source chunk is in the top-k.\n"
        "  TP = positive query, chunk found       FN = positive query, chunk missed\n"
        "  FP = hard-negative query, chunk found  TN = hard-negative query, chunk absent\n"
        "precision = TP/(TP+FP)   recall = TP/(TP+FN)   f1 = harmonic mean of the two\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eval_dataset.jsonl")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--raw-out", default="eval_raw_results.csv", help="Per-query raw results, for spot-checking")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    if not dataset:
        raise SystemExit(f"No rows loaded from {args.dataset}")
    logger.info("Loaded %d eval rows", len(dataset))

    store = build_store()

    results: list[RowResult] = []
    for i, row in enumerate(dataset, start=1):
        chunk_id = row["chunk_id"]
        doc_type = row["doc_type"]

        for query_type, query in (
            ("positive", row["positive_query"]),
            ("hard_negative", row["hard_negative_query"]),
        ):
            dense_ids = dense_search_ids(store, query, args.k)
            hybrid_ids = hybrid_search_ids(store, query, args.k)

            results.append(
                RowResult(chunk_id, doc_type, query_type, "dense", rank_of(chunk_id, dense_ids), args.k)
            )
            results.append(
                RowResult(chunk_id, doc_type, query_type, "hybrid", rank_of(chunk_id, hybrid_ids), args.k)
            )

        if i % 20 == 0 or i == len(dataset):
            logger.info("Evaluated %d/%d rows", i, len(dataset))

    summary = summarize(results, args.k)
    print_report(summary, args.k)

    overall = summarize_overall(results)
    print_overall_report(overall, args.k)

    raw_path = Path(args.raw_out)
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "doc_type", "query_type", "mode", "rank", "k"])
        for r in results:
            writer.writerow([r.chunk_id, r.doc_type, r.query_type, r.mode, r.rank if r.rank else "", r.k])
    logger.info("Wrote per-query raw results to %s", raw_path)


if __name__ == "__main__":
    main()