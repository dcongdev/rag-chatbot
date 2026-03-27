# AI Knowledge Base Assistant

A lightweight Retrieval-Augmented Generation (RAG) application that enables users to ask questions about their own documents using natural language. The system combines semantic search with Google's Gemini API to generate context-aware answers grounded in uploaded knowledge sources.

## Features

- **Document Management:** Upload, view, replace, and delete PDF/TXT documents.
- **Document Processing:** Extract text, split documents into overlapping chunks, and generate embeddings.
- **Semantic Search:** Retrieve relevant document chunks using Sentence Transformers and ChromaDB.
- **AI Question Answering:** Generate answers based on retrieved context using Gemini API.
- **Source References:** Display supporting document sources alongside AI-generated answers.
- **Interactive Interface:** Streamlit-based chat interface with session history and document management.

## Tech Stack

- **Language:** Python
- **Framework:** LangChain
- **LLM:** Google Gemini API
- **Embeddings:** Sentence Transformers
- **Vector Database:** ChromaDB
- **Interface:** Streamlit
- **Environment:** uv

## How It Works

1. Users upload PDF or TXT documents to the knowledge base.
2. Documents are parsed, split into chunks, and converted into vector embeddings.
3. Embeddings and document metadata are stored in ChromaDB.
4. User questions are embedded and matched against relevant document chunks.
5. Retrieved context is sent to Gemini to generate an answer with source references.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

Clone the repository:

```bash
git clone https://github.com/dcongdev/rag-chatbot.git
cd rag-chatbot
```

Install dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

### Run the Application

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Testing

Run the automated test suite:

```bash
uv run python -m unittest discover -s tests -v
```

## Notes

- Document embeddings are stored locally using persistent ChromaDB storage.
- Gemini API requires an internet connection and a valid API key.
- Chat history is maintained within the current Streamlit session.
- Local data, API credentials, and runtime files are excluded from version control.