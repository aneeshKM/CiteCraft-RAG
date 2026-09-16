"""Streamlit interface for CiteCraft RAG."""

from __future__ import annotations

import hashlib
import logging
import sys
import tempfile
from pathlib import Path

# Support `streamlit run app.py` before the project is installed as a package.
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import streamlit as st  # noqa: E402

from citecraft import RagService, Settings  # noqa: E402

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="CiteCraft RAG",
    page_icon="📚",
    layout="wide",
)


def get_service(settings: Settings) -> RagService:
    signature = hashlib.sha256(
        (
            settings.openai_api_key
            + settings.postgres_url
            + settings.redis_url
            + settings.collection_name
        ).encode("utf-8")
    ).hexdigest()
    if st.session_state.get("service_signature") != signature:
        st.session_state.rag_service = RagService(settings)
        st.session_state.service_signature = signature
    return st.session_state.rag_service


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("active_hashes", [])
    st.session_state.setdefault("indexed_files", {})


def index_uploads(service: RagService, uploads: list) -> None:
    service.check_connections()
    with tempfile.TemporaryDirectory(prefix="citecraft-") as temp_dir:
        for upload in uploads:
            safe_name = Path(upload.name).name
            temp_path = Path(temp_dir) / safe_name
            temp_path.write_bytes(upload.getbuffer())
            with st.status(f"Indexing {safe_name}…", expanded=True) as status:
                result = service.ingest(temp_path, filename=safe_name)
                cache_note = "reused existing index" if result.was_cached else "created new index"
                st.write(f"{result.element_count} elements · {cache_note}")
                status.update(label=f"Ready: {safe_name}", state="complete", expanded=False)
            if result.file_hash not in st.session_state.active_hashes:
                st.session_state.active_hashes.append(result.file_hash)
            st.session_state.indexed_files[result.file_hash] = safe_name


def render_sidebar(base_settings: Settings) -> Settings:
    with st.sidebar:
        st.header("Document workspace")
        st.caption(
            "Extracted content stays in local Redis/PGVector; relevant excerpts are sent "
            "to OpenAI for summaries and answers."
        )
        api_key = base_settings.openai_api_key or st.text_input(
            "OpenAI API key",
            type="password",
            help="Used for embeddings, retrieval summaries, and answers.",
        )
        settings = Settings.from_env(api_key=api_key)
        uploads = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Text, tables, and page metadata will be indexed.",
        )

        if st.button("Index documents", type="primary", use_container_width=True):
            if not api_key:
                st.error("Add an OpenAI API key first.")
            elif not uploads:
                st.warning("Choose at least one PDF.")
            else:
                try:
                    index_uploads(get_service(settings), uploads)
                    st.success("Documents are ready for questions.")
                except Exception as exc:
                    logging.exception("Document indexing failed")
                    st.error(f"Indexing failed: {exc}")

        if st.session_state.indexed_files:
            st.divider()
            st.subheader("Active sources")
            for filename in st.session_state.indexed_files.values():
                st.caption(f"✓ {filename}")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(
            f"Model: `{settings.chat_model}`  \n"
            f"Retrieval depth: `{settings.top_k}`  \n"
            f"Parser: `{settings.partition_strategy}`"
        )
    return settings


def render_message(message: dict[str, object]) -> None:
    with st.chat_message(str(message["role"])):
        st.markdown(str(message["content"]))
        sources = message.get("sources", [])
        if sources:
            with st.expander("Retrieved sources"):
                for source in sources:
                    st.write(source)
        if latency := message.get("latency"):
            st.caption(f"Answered in {float(latency):.2f}s")


def main() -> None:
    initialize_state()
    base_settings = Settings.from_env()
    settings = render_sidebar(base_settings)

    st.title("CiteCraft RAG")
    st.write("Ask questions across your PDFs and trace every answer back to its source page.")

    if not st.session_state.active_hashes:
        st.info("Upload and index one or more PDFs from the sidebar to begin.")

    for message in st.session_state.messages:
        render_message(message)

    if prompt := st.chat_input(
        "Ask a question about the indexed documents…",
        disabled=not bool(st.session_state.active_hashes),
    ):
        prior_history = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in st.session_state.messages
        ]
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        render_message(user_message)

        with (
            st.chat_message("assistant"),
            st.spinner("Retrieving evidence and composing an answer…"),
        ):
            try:
                answer = get_service(settings).answer(
                    prompt,
                    allowed_hashes=st.session_state.active_hashes,
                    chat_history=prior_history,
                )
                source_labels = [source.label for source in answer.sources]
                assistant_message = {
                    "role": "assistant",
                    "content": answer.text,
                    "sources": source_labels,
                    "latency": answer.latency_seconds,
                }
                st.session_state.messages.append(assistant_message)
                st.markdown(answer.text)
                if source_labels:
                    with st.expander("Retrieved sources"):
                        for label in source_labels:
                            st.write(label)
                st.caption(f"Answered in {answer.latency_seconds:.2f}s")
            except Exception as exc:
                logging.exception("Answer generation failed")
                st.error(f"Could not answer the question: {exc}")


if __name__ == "__main__":
    main()
