"""Web UI: upload data → LiteParse → chunk → index (Qdrant) → agentic RAG chat."""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
from chunking import (  # noqa: E402
    DEFAULT_CHUNKER_CONFIG_PATH,
    chunk_file,
    ensure_chunker_config,
    load_chunker,
)
from data_pipeline import (  # noqa: E402
    build_store,
    create_index,
    index_status,
    ingest_paths,
    list_documents,
    load_model_config,
    remove_document,
    reset_index,
)
from parsing import SUPPORTED_FORMATS, parse_file  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = ROOT / "data" / "uploads"
PREVIEW_CHARS = 4_000
CHUNK_SAMPLE = 5
CHUNK_PREVIEW_CHARS = 600
NO_SOURCES = "_Sources will appear here._"


# ───────────────────────── ingestion ─────────────────────────

def _chunker_config_path() -> str:
    p = ROOT / config.CHUNKER_CONFIG_PATH
    if not p.is_file():
        p = ROOT / DEFAULT_CHUNKER_CONFIG_PATH
    return str(p)


def _save_inputs(files: list[str] | None, pasted_text: str) -> tuple[list[Path], str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    notes: list[str] = []

    if pasted_text and pasted_text.strip():
        dest = UPLOAD_DIR / f"pasted_{int(time.time())}.md"
        dest.write_text(pasted_text.strip(), encoding="utf-8")
        paths.append(dest)
        notes.append(f"Saved pasted text as `{dest.name}`")

    for raw in files or []:
        src = Path(raw)
        if not src.is_file():
            continue
        dest = UPLOAD_DIR / src.name
        if dest.exists() and dest.resolve() != src.resolve():
            dest = UPLOAD_DIR / f"{dest.stem}_{int(time.time())}{dest.suffix}"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        paths.append(dest)
        notes.append(f"Staged `{dest.name}`")

    if not paths:
        return [], "Add at least one file or paste text to process."
    return paths, "\n".join(notes)


def run_ingest_pipeline(files, pasted_text, force_reindex, model_choice):
    """Generator: yields (parse_preview, chunk_samples, log_markdown, status) after each step."""
    use_quantized = model_choice == "Quantized (Q4, faster, lower memory)"
    paths, staging_msg = _save_inputs(files, pasted_text)
    if not paths:
        yield "", "", staging_msg, staging_msg
        return

    chunker_path = _chunker_config_path()
    ensure_chunker_config(chunker_path)
    chunker = load_chunker(chunker_path)

    log: list[str] = [staging_msg, ""]
    parse_blocks: list[str] = []
    chunk_blocks: list[str] = []

    def snap(status: str):
        return "\n\n".join(parse_blocks), "\n\n".join(chunk_blocks), "\n".join(log), status

    yield snap("Working…")

    for path in paths:
        log.append(f"## {path.name}")
        ext = path.suffix.lower().lstrip(".")
        parser_label = "LiteParse" if ext in {"pdf", "csv", "xlsx", "xls", "docx", "doc"} else "plain text"
        log.append(f"**1. Parsing** ({parser_label})…")
        yield snap("Parsing…")

        try:
            text = parse_file(path)
        except Exception as exc:
            logger.exception("Parse failed")
            log.append(f"❌ Parse failed for {path.name}: {exc}")
            yield snap("Parse failed")
            return

        preview = text[:PREVIEW_CHARS] + ("\n\n… (truncated)" if len(text) > PREVIEW_CHARS else "")
        parse_blocks.append(f"### {path.name} — {len(text):,} characters\n\n{preview}")
        log.append(f"   → {len(text):,} characters extracted.")

        log.append("**2. Chunking** (recursive)…")
        yield snap("Chunking…")
        try:
            document = chunk_file(path, chunker)
        except Exception as exc:
            logger.exception("Chunking failed")
            log.append(f"❌ Chunking failed for {path.name}: {exc}")
            yield snap("Chunking failed")
            return

        n = len(document.chunks)
        sample_text = "\n\n---\n\n".join(
            f"**Chunk {i + 1}** ({len(c.text)} chars)\n{c.text[:CHUNK_PREVIEW_CHARS]}"
            + ("…" if len(c.text) > CHUNK_PREVIEW_CHARS else "")
            for i, c in enumerate(document.chunks[:CHUNK_SAMPLE])
        )
        chunk_blocks.append(f"### {path.name} — {n} chunk(s)\n\n{sample_text}")
        log.append(f"   → {n} chunk(s).")

    model_type = "quantized (Q4)" if use_quantized else "unquantized (full precision)"
    log += ["", f"**3. Indexing** (ONNX embeddings + Qdrant hybrid, using {model_type} model)… this can take a while."]
    yield snap("Indexing…")
    try:
        count = ingest_paths([str(p) for p in paths], chunker_config=chunker_path, force=force_reindex, use_quantized=use_quantized)
    except Exception as exc:
        logger.exception("Indexing failed")
        log.append(f"❌ Indexing failed: {exc}")
        yield snap("Indexing failed")
        return

    log += [
        f"   → Wrote **{count}** chunk(s) to collection `{config.COLLECTION_NAME}`.",
        "",
        "✅ Ready. Open the **Chat** tab to ask questions about your documents.",
    ]
    yield snap(f"Indexed {count} chunk(s) from {len(paths)} source(s).")


# ───────────────────────── index management ─────────────────────────

def _docs_table_rows(docs: list[dict]) -> list[list]:
    return [
        [d["source"], d.get("file_type", ""), d["chunks"], d.get("ingested_at", "")]
        for d in docs
    ]


def refresh_documents():
    """Return (table rows, updated dropdown, status) for the Documents tab."""
    try:
        docs = list_documents()
    except Exception as exc:
        logger.exception("Listing documents failed")
        return (
            [],
            gr.Dropdown(choices=[], value=None),
            f"⚠️ Could not reach Qdrant at `{config.QDRANT_URL}`: {exc}",
        )
    return (
        _docs_table_rows(docs),
        gr.Dropdown(choices=[d["source"] for d in docs], value=None),
        f"{len(docs)} document(s) in the index.",
    )


def remove_document_ui(source, delete_file):
    if not source:
        rows, dropdown, _ = refresh_documents()
        return rows, dropdown, "Select a document to remove first."
    try:
        msg = remove_document(source, delete_file=delete_file)
    except Exception as exc:
        logger.exception("Remove failed")
        rows, dropdown, _ = refresh_documents()
        return rows, dropdown, f"❌ Remove failed: {exc}"
    rows, dropdown, status = refresh_documents()
    return rows, dropdown, f"✅ {msg} {status}"


def show_index_status():
    try:
        status = index_status()
    except Exception as exc:
        logger.exception("Status check failed")
        return f"⚠️ Could not reach Qdrant at `{config.QDRANT_URL}`: {exc}"
    if not status["exists"]:
        return (
            f"Collection `{config.COLLECTION_NAME}` does not exist yet — "
            "click **Create / verify collection** or run the ingest pipeline."
        )
    lines = [
        f"Collection `{config.COLLECTION_NAME}`: "
        f"**{status['points']}** point(s) across **{len(status['documents'])}** document(s)."
    ]
    lines += [f"- `{d['source']}` — {d['chunks']} chunk(s)" for d in status["documents"]]
    return "\n".join(lines)


def create_index_ui():
    try:
        # Update config to use the saved model choice
        use_quantized = load_model_config()
        config.EMBEDDING_MODEL_PATH = config.EMBEDDING_MODEL_PATH_QUANTIZED if use_quantized else config.EMBEDDING_MODEL_PATH_UNQUANTIZED
        return "✅ " + create_index()
    except Exception as exc:
        logger.exception("Create index failed")
        return f"❌ Failed: {exc}"


def reset_index_ui(confirm):
    if not confirm:
        return "Tick the confirmation checkbox first — reset deletes **every** indexed chunk."
    try:
        return "✅ " + reset_index()
    except Exception as exc:
        logger.exception("Reset index failed")
        return f"❌ Failed: {exc}"


def refresh_system_info():
    """Refresh system info to show current model choice."""
    use_quantized = load_model_config()
    model_status = "Quantized (Q4)" if use_quantized else "Unquantized (full precision)"
    return (
        f"- **Qdrant:** `{config.QDRANT_URL}` / `{config.COLLECTION_NAME}`\n"
        f"- **LLM:** `{config.LLM_BASE_URL}`\n"
        f"- **Embeddings:** {model_status}\n"
        f"- **Upload directory:** `{UPLOAD_DIR}`\n"
    )


# ───────────────────────── agentic chat ─────────────────────────

def format_sources(docs: list[str]) -> str:
    if not docs:
        return "_No relevant sources found._"
    blocks = []
    for d in docs:
        header, _, body = d.partition("\n")
        blocks.append(
            f"<details><summary>{header[:120]}</summary>\n\n{body.strip()[:1500]}\n\n</details>"
        )
    return "\n\n".join(blocks)


def respond(message, history):
    message = (message or "").strip()
    history = history or []
    if not message:
        yield history, "", NO_SOURCES
        return

    from agent import app as agent_app   # lazy: keeps the Ingest tab fast to start

    history.append({"role": "user", "content": message})
    trace_idx = len(history)
    history.append({
        "role": "assistant",
        "content": "Starting…",
        "metadata": {"title": "🧠 Agent steps", "status": "pending"},
    })
    yield history, "", "_Searching…_"

    steps, sources_md, answer = [], "_Searching…_", ""
    state_in = {
        "question": message,
        "retrieval_attempts": 0,
        "documents": [],
        "graded_relevant": False,
    }

    try:
        for update in agent_app.stream(state_in, stream_mode="updates"):
            for node, ch in update.items():
                if node == "retrieve":
                    steps.append(f"📥 **Retrieve**: {len(ch['documents'])} chunks found")
                elif node == "grade_documents":
                    steps.append(f"⚖️ **Grade**: {len(ch['documents'])} relevant chunk(s) kept")
                    sources_md = format_sources(ch["documents"])
                elif node == "rewrite":
                    steps.append(f"✏️ **Rewrite** (attempt {ch['retrieval_attempts']}): _{ch['question']}_")
                elif node == "generate":
                    steps.append("✅ **Generate**: answer ready")
                    answer = ch["answer"]
                history[trace_idx]["content"] = "\n\n".join(steps)
                yield history, "", sources_md
    except Exception as exc:
        logger.exception("Chat failed")
        answer = (
            f"⚠️ Error talking to the LLM or retriever: {exc}\n\n"
            f"Check that Qdrant (`{config.QDRANT_URL}`) and the LLM (`{config.LLM_BASE_URL}`) are running."
        )

    history[trace_idx]["metadata"]["status"] = "done"
    history.append({"role": "assistant", "content": answer or "_No answer produced._"})
    yield history, "", sources_md


# ───────────────────────── UI ─────────────────────────

def build_app() -> gr.Blocks:
    formats = ", ".join(sorted(SUPPORTED_FORMATS))
    with gr.Blocks(title="Agentic RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 📚 Agentic RAG\n"
            "Upload or paste content, run **parse → chunk → index**, then chat. "
            "The agent retrieves, grades, rewrites the query if needed, and answers with citations."
        )

        # ---------- Ingest ----------
        with gr.Tab("1 · Ingest"):
            gr.Markdown(
                f"Supported file types: **{formats}**. PDFs and office files go through "
                "**LiteParse**; `.txt` / `.md` are read as UTF-8 text.\n\n"
                "**Note:** The embedding model choice will be saved and used for retrieval. "
                "If you switch models, you should reset the index and re-index all documents."
            )
            with gr.Row():
                file_in = gr.File(label="Upload documents", file_count="multiple", type="filepath")
                text_in = gr.Textbox(
                    label="Or paste text", lines=8,
                    placeholder="Paste notes or articles here (saved as Markdown and indexed).",
                )
            force_cb = gr.Checkbox(
                label="Force re-index (ignore manifest, replace chunks for changed files)", value=False
            )
            model_choice = gr.Radio(
                choices=["Quantized (Q4, faster, lower memory)", "Unquantized (full precision, better quality)"],
                value="Quantized (Q4, faster, lower memory)",
                label="Embedding model",
                interactive=True
            )
            run_btn = gr.Button("Run pipeline: Parse → Chunk → Index", variant="primary")
            status_out = gr.Textbox(label="Status", interactive=False)
            with gr.Accordion("Pipeline log", open=True):
                log_out = gr.Markdown()
            with gr.Row():
                parse_out = gr.Textbox(label="Parse preview", lines=14, max_lines=24, interactive=False)
                chunk_out = gr.Textbox(label="Chunk samples", lines=14, max_lines=24, interactive=False)

            ingest_event = run_btn.click(
                run_ingest_pipeline,
                inputs=[file_in, text_in, force_cb, model_choice],
                outputs=[parse_out, chunk_out, log_out, status_out],
            )

        # ---------- Documents (index management) ----------
        with gr.Tab("2 · Documents"):
            gr.Markdown(
                "Documents currently in the vector index. You can **remove a single document** "
                "(e.g. one of two uploaded books) without re-ingesting the rest."
            )
            refresh_docs_btn = gr.Button("🔄 Refresh list")
            docs_table = gr.Dataframe(
                headers=["source", "type", "chunks", "ingested_at"],
                value=[],
                interactive=False,
                label="Indexed documents",
            )
            with gr.Row():
                doc_dropdown = gr.Dropdown(
                    label="Document to remove", choices=[], interactive=True, scale=3
                )
                delete_file_cb = gr.Checkbox(
                    label="Also delete the uploaded file from data/uploads", value=False, scale=2
                )
            remove_btn = gr.Button("🗑 Remove selected document", variant="stop")
            docs_status = gr.Textbox(label="Status", interactive=False)

            refresh_docs_btn.click(
                refresh_documents, outputs=[docs_table, doc_dropdown, docs_status]
            )
            remove_btn.click(
                remove_document_ui,
                inputs=[doc_dropdown, delete_file_cb],
                outputs=[docs_table, doc_dropdown, docs_status],
            )
            # Keep the list in sync after a successful ingest run.
            ingest_event.then(refresh_documents, outputs=[docs_table, doc_dropdown, docs_status])

        # ---------- Chat ----------
        with gr.Tab("3 · Chat"):
            gr.Markdown(f"Hybrid retrieval over collection `{config.COLLECTION_NAME}`.")
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(type="messages", height=540, label="Conversation")
                    chat_in = gr.Textbox(
                        placeholder="Ask about your indexed documents…", show_label=False, autofocus=True
                    )
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")
                    gr.Examples(
                        ["Summarize the main topics in my uploaded documents.",
                         "What entities or organizations are mentioned?"],
                        inputs=chat_in,
                    )
                with gr.Column(scale=2):
                    gr.Markdown("### 📎 Relevant sources (this question)")
                    sources = gr.Markdown(NO_SOURCES)

            for trigger in (chat_in.submit, send_btn.click):
                trigger(respond, [chat_in, chatbot], [chatbot, chat_in, sources])
            clear_btn.click(lambda: ([], "", NO_SOURCES), None, [chatbot, chat_in, sources])

        # ---------- System ----------
        with gr.Tab("4 · System"):
            from data_pipeline import load_model_config
            use_quantized = load_model_config()
            model_status = "Quantized (Q4)" if use_quantized else "Unquantized (full precision)"
            system_info = gr.Markdown(
                f"- **Qdrant:** `{config.QDRANT_URL}` / `{config.COLLECTION_NAME}`\n"
                f"- **LLM:** `{config.LLM_BASE_URL}`\n"
                f"- **Embeddings:** {model_status}\n"
                f"- **Upload directory:** `{UPLOAD_DIR}`\n"
            )
            warm_btn = gr.Button("Warm up index connection (load embedder)")
            warm_out = gr.Textbox(label="Warm-up", interactive=False)

            def warm_up() -> str:
                try:
                    build_store(use_quantized=load_model_config())
                    return "Embedding model and Qdrant client loaded."
                except Exception as exc:
                    return f"Warm-up failed: {exc}"

            warm_btn.click(warm_up, outputs=warm_out)

            def warm_up() -> str:
                try:
                    build_store(use_quantized=load_model_config())
                    return "Embedding model and Qdrant client loaded."
                except Exception as exc:
                    return f"Warm-up failed: {exc}"

            warm_btn.click(warm_up, outputs=warm_out)

            gr.Markdown(
                "#### Vector database\n"
                "Create the collection explicitly, check what's inside, or wipe it entirely. "
                "*(Create/reset load the embedding model, so they can take a moment.)*"
            )
            with gr.Row():
                check_db_btn = gr.Button("Check status")
                create_db_btn = gr.Button("Create / verify collection", variant="primary")
            reset_confirm_cb = gr.Checkbox(
                label="I understand reset deletes EVERY indexed chunk and clears the manifest",
                value=False,
            )
            reset_db_btn = gr.Button("⚠️ Reset index (delete all documents)", variant="stop")
            db_status_out = gr.Textbox(label="Vector DB status", interactive=False)

            check_db_btn.click(show_index_status, outputs=db_status_out)
            create_db_btn.click(create_index_ui, outputs=db_status_out).then(
                refresh_documents, outputs=[docs_table, doc_dropdown, docs_status]
            ).then(
                refresh_system_info, outputs=system_info
            )
            def _refresh_docs_and_uncheck():
                rows, dropdown, status = refresh_documents()
                return rows, dropdown, status, False

            reset_db_btn.click(
                reset_index_ui, inputs=reset_confirm_cb, outputs=db_status_out
            ).then(
                _refresh_docs_and_uncheck,
                outputs=[docs_table, doc_dropdown, docs_status, reset_confirm_cb],
            ).then(
                refresh_system_info, outputs=system_info
            )

        # Populate the Documents tab on first page load.
        demo.load(refresh_documents, outputs=[docs_table, doc_dropdown, docs_status])
        demo.load(show_index_status, outputs=db_status_out)
        demo.load(refresh_system_info, outputs=system_info)

    return demo


def main() -> None:
    import os
    os.chdir(ROOT)
    build_app().queue().launch(server_name="127.0.0.1", server_port=7060)


if __name__ == "__main__":
    main()