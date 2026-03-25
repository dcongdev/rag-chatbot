"""Session-only document chat and Knowledge Base management interface."""

from pathlib import PurePosixPath

import streamlit as st

from knowledge_base.generation import GenerationError, answer_question
from knowledge_base.retrieval import retrieve
from knowledge_base.storage import ChromaStore
from knowledge_base.uploads import index_upload, replace_upload


@st.cache_resource(show_spinner=False, scope="session", on_release=lambda store: store.close())
def get_store() -> ChromaStore:
    """Reuse the database client and lazy embedding model across session reruns."""
    return ChromaStore()


def clear_results() -> None:
    for key in ("search_results", "searched_query"):
        st.session_state.pop(key, None)
    st.session_state["chat_messages"] = []


def clear_chat() -> None:
    st.session_state["chat_messages"] = []


def clear_delete_confirmation() -> None:
    st.session_state.pop("pending_delete", None)


def manage_documents() -> None:
    """Render current storage metadata and explicit replacement/deletion actions."""
    message = st.session_state.pop("management_success", None)
    if message:
        st.success(message)
    try:
        store = get_store()
        documents = store.list_documents()
    except Exception as error:
        st.error(f"Could not list indexed documents: {error}")
        return
    if not documents:
        st.info("No indexed documents yet. Add PDF or TXT files above.")
        return
    st.dataframe(
        [{"Filename": document.filename, "Chunks": document.chunk_count} for document in documents],
        hide_index=True,
    )
    selected = st.selectbox(
        "Manage a document", [document.filename for document in documents],
        key="managed_document", on_change=clear_delete_confirmation,
    )
    replacement = st.file_uploader(
        "Replacement PDF or TXT", type=["pdf", "txt"], key=f"replacement:{selected}",
    )
    st.caption("Replacement keeps the selected filename and replaces all of its indexed content.")
    if st.button("Replace document", key="replace", disabled=replacement is None):
        clear_results()
        clear_delete_confirmation()
        try:
            with st.spinner(f"Replacing {selected}..."):
                ids = replace_upload(selected, replacement.name, replacement.getvalue(), store)
        except Exception as error:
            st.error(f"Could not replace {selected}: {error}")
        else:
            st.session_state["management_success"] = f"Replaced {selected}: {len(ids)} chunks indexed."
            st.rerun()
    if st.button("Delete document", key="request_delete"):
        st.session_state["pending_delete"] = selected
    if st.session_state.get("pending_delete") == selected:
        st.warning(f"Delete {selected} and all of its chunks? This cannot be undone.")
        if st.button("Confirm deletion", key="confirm_delete"):
            clear_results()
            try:
                with st.spinner(f"Deleting {selected}..."):
                    count = store.delete_document(selected)
            except Exception as error:
                st.error(f"Could not delete {selected}: {error}")
            else:
                clear_delete_confirmation()
                st.session_state["management_success"] = f"Deleted {selected} and {count} chunks."
                st.rerun()
        if st.button("Cancel", key="cancel_delete"):
            clear_delete_confirmation()
            st.rerun()


def knowledge_base_tab() -> None:
    st.caption("Add and manage PDF/TXT documents. Changes clear displayed chat and search results.")
    uploads = st.file_uploader(
        "PDF and TXT files", type=["pdf", "txt"], accept_multiple_files=True,
        key="uploads",
    )
    if st.button("Index documents", disabled=not uploads, key="index"):
        clear_results()
        clear_delete_confirmation()
        try:
            with st.spinner("Opening the knowledge base..."):
                store = get_store()
        except Exception as error:
            st.error(f"Could not open the knowledge base: {error}")
        else:
            for upload in uploads:
                filename = PurePosixPath(upload.name.replace("\\", "/")).name
                try:
                    # A prior upload may have saved chunks before reporting a failure.
                    existing = {document.filename for document in store.list_documents()}
                    if filename in existing:
                        st.info(f"{filename}: already indexed. Use Replace document to update its content.")
                        continue
                    with st.spinner(f"Indexing {upload.name}..."):
                        ids = index_upload(upload.name, upload.getvalue(), store)
                except Exception as error:
                    st.error(
                        f"Could not index {upload.name}: {error}. "
                        "Other files can still be processed. If some chunks were saved, "
                        "retrying will skip them."
                    )
                else:
                    if ids:
                        st.success(f"{upload.name}: indexed {len(ids)} new chunks.")
                    else:
                        st.info(
                            f"{upload.name}: no new chunks. The file is empty, has no "
                            "extractable text, or is already indexed."
                        )

    manage_documents()


def semantic_search() -> None:
    with st.form("search"):
        query = st.text_input("Search query", key="query")
        top_k = st.number_input(
            "Number of results", min_value=1, max_value=20, value=4, step=1,
            key="top_k",
        )
        submitted = st.form_submit_button("Search documents", key="search_submit")
    if submitted:
        st.session_state.pop("search_results", None)
        st.session_state.pop("searched_query", None)
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            try:
                with st.spinner("Searching documents..."):
                    results = retrieve(query, get_store(), top_k=int(top_k))
            except Exception as error:
                st.error(f"Could not search the knowledge base: {error}")
            else:
                st.session_state["search_results"] = results
                st.session_state["searched_query"] = query.strip()

    if "search_results" in st.session_state:
        results = st.session_state["search_results"]
        if not results:
            st.info("No matching chunks found. Index documents with extractable text first.")
        else:
            st.subheader("Retrieved chunks")
            st.text(f"Results for: {st.session_state['searched_query']}")
            st.caption("Higher similarity scores indicate closer matches, not answer confidence.")
            for position, chunk in enumerate(results, start=1):
                with st.container(border=True):
                    st.caption(f"Chunk {position}")
                    st.text(f"Source: {chunk.metadata.get('source', 'Unknown')}")
                    if chunk.metadata.get("page") is not None:
                        st.caption(f"PDF page: {chunk.metadata['page']}")
                    if chunk.similarity_score is not None:
                        st.caption(f"Similarity score: {chunk.similarity_score:.3f}")
                    st.text(chunk.content)


def display_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("kind") == "error":
            st.error(message["content"])
        elif message.get("kind") == "info":
            st.info(message["content"])
        elif message["role"] == "user":
            st.text(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources")
            for source in message["sources"]:
                label = f"[{source.id}] {source.filename}"
                if source.page is not None:
                    label += f" — PDF page {source.page}"
                st.text(label)


def chat_tab() -> None:
    st.caption("Ask about your documents. History is shown only in this session; each question is answered independently.")
    controls, actions = st.columns([3, 1])
    with controls:
        top_k = st.number_input(
            "Passages to use", min_value=1, max_value=20, value=4, step=1,
            key="answer_top_k",
        )
    with actions:
        st.button("Clear Chat", key="clear_chat", on_click=clear_chat)
    history = st.session_state.setdefault("chat_messages", [])
    transcript = st.container()
    with transcript:
        if not history:
            st.caption("Add documents in Knowledge Base, then ask a question here.")
        for message in history:
            display_message(message)
    question = st.chat_input("Ask about your documents", key="question")
    if question is not None:
        if not question.strip():
            st.warning("Enter a question.")
        else:
            message = {"role": "user", "content": question.strip()}
            history.append(message)
            with transcript:
                display_message(message)
                try:
                    with st.spinner("Finding evidence and asking Gemini..."):
                        answer = answer_question(question.strip(), get_store(), top_k=int(top_k))
                except GenerationError as error:
                    reply = {"role": "assistant", "content": str(error), "kind": "error"}
                except Exception:
                    reply = {"role": "assistant", "content": "Could not answer the question. Check the knowledge base and retry.", "kind": "error"}
                else:
                    reply = {
                        "role": "assistant", "content": answer.answer,
                        "sources": answer.sources,
                        "kind": "answer" if answer.has_answer else "info",
                    }
                history.append(reply)
                display_message(reply)
    with st.expander("Semantic search (optional)", expanded=False):
        semantic_search()


def main() -> None:
    st.set_page_config(page_title="AI Knowledge Base Assistant")
    st.title("AI Knowledge Base Assistant")
    ai_chat, knowledge_base = st.tabs(["AI Chat", "Knowledge Base"])
    # Process mutations before rendering chat so removed evidence is never displayed.
    with knowledge_base:
        knowledge_base_tab()
    with ai_chat:
        chat_tab()


if __name__ == "__main__":
    main()
