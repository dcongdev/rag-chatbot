"""Minimal local document indexing and semantic search interface."""

import streamlit as st

from knowledge_base.generation import GenerationError, answer_question
from knowledge_base.retrieval import retrieve
from knowledge_base.storage import ChromaStore
from knowledge_base.uploads import index_upload


@st.cache_resource(show_spinner=False, scope="session", on_release=lambda store: store.close())
def get_store() -> ChromaStore:
    """Reuse the database client and lazy embedding model across session reruns."""
    return ChromaStore()


def main() -> None:
    st.set_page_config(page_title="AI Knowledge Base Assistant")
    st.title("AI Knowledge Base Assistant")
    st.write("Upload documents, then search their contents for relevant passages.")

    st.subheader("Add documents")
    uploads = st.file_uploader(
        "PDF and TXT files", type=["pdf", "txt"], accept_multiple_files=True,
        key="uploads",
    )
    if st.button("Index documents", disabled=not uploads, key="index"):
        st.session_state.pop("search_results", None)
        st.session_state.pop("searched_query", None)
        st.session_state.pop("question_answer", None)
        st.session_state.pop("answered_question", None)
        try:
            with st.spinner("Opening the knowledge base..."):
                store = get_store()
        except Exception as error:
            st.error(f"Could not open the knowledge base: {error}")
        else:
            for upload in uploads:
                try:
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

    st.subheader("Search documents")
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
                with st.expander(f"Chunk {position}", expanded=True):
                    st.text(f"Source: {chunk.metadata.get('source', 'Unknown')}")
                    if chunk.metadata.get("page") is not None:
                        st.caption(f"PDF page: {chunk.metadata['page']}")
                    if chunk.similarity_score is not None:
                        st.caption(f"Similarity score: {chunk.similarity_score:.3f}")
                    st.text(chunk.content)

    st.subheader("Ask your documents")
    st.caption("Gemini answers using relevant passages sent from your indexed documents.")
    with st.form("question_answering"):
        question = st.text_input("Question", key="question")
        answer_top_k = st.number_input(
            "Passages to use", min_value=1, max_value=20, value=4, step=1,
            key="answer_top_k",
        )
        asked = st.form_submit_button("Get answer", key="ask_submit")
    if asked:
        st.session_state.pop("question_answer", None)
        st.session_state.pop("answered_question", None)
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                with st.spinner("Finding evidence and asking Gemini..."):
                    answer = answer_question(question, get_store(), top_k=int(answer_top_k))
            except GenerationError as error:
                st.error(str(error))
            except Exception:
                st.error("Could not answer the question. Check the knowledge base and retry.")
            else:
                st.session_state["question_answer"] = answer
                st.session_state["answered_question"] = question.strip()
    if "question_answer" in st.session_state:
        answer = st.session_state["question_answer"]
        st.text(f"Question: {st.session_state['answered_question']}")
        if not answer.has_answer:
            st.info(answer.answer)
        else:
            st.markdown(answer.answer)
            st.write("Sources")
            for source in answer.sources:
                label = f"[{source.id}] {source.filename}"
                if source.page is not None:
                    label += f" — PDF page {source.page}"
                st.text(label)


if __name__ == "__main__":
    main()
