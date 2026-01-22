# RAG Document Assistant

A local RAG (Retrieval-Augmented Generation) powered document assistant that lets you chat with your documents using a local LLM.

## Features

- **Multi-format document ingestion**: PDF, DOCX, Markdown, code files, and plain text
- **Local LLM**: Uses Ollama for private, offline inference
- **Vector search**: ChromaDB with sentence-transformers embeddings
- **Multiple interfaces**: Gradio web UI, FastAPI REST API, and CLI
- **Source citations**: Answers include references to source documents

## Requirements

- Python 3.13+
- [Ollama](https://ollama.ai/) with a model installed (default: llama3.2)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jadfarhat-cell/Document-Assistant.git
cd Document-Assistant
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. Install Ollama and pull a model:
```bash
ollama pull llama3.2
```

4. (Optional) Copy and configure environment variables:
```bash
cp .env.example .env
```

## Usage

### Web UI (Gradio)

```bash
python cli.py serve --ui
```

Open http://localhost:7861 in your browser.

### REST API (FastAPI)

```bash
python cli.py serve --api
```

API docs available at http://localhost:8000/docs

### CLI

```bash
# Ingest a document
python cli.py ingest document.pdf

# Ingest a directory
python cli.py ingest ./docs --directory

# Query your documents
python cli.py query "What is the main topic?"

# View system stats
python cli.py stats

# Clear all indexed documents
python cli.py clear
```

## Configuration

Configure via environment variables (prefix with `RAG_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_OLLAMA_MODEL` | llama3.2 | Ollama model to use |
| `RAG_OLLAMA_BASE_URL` | http://localhost:11434 | Ollama API URL |
| `RAG_CHUNK_SIZE` | 512 | Text chunk size |
| `RAG_CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `RAG_TOP_K` | 5 | Number of documents to retrieve |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│   │ Document │───>│   Loader     │───>│   Chunker    │───>│  Embedder   │  │
│   │ (PDF,    │    │ (Extract     │    │ (Split into  │    │ (Generate   │  │
│   │  DOCX,   │    │  text)       │    │  512-token   │    │  vectors)   │  │
│   │  MD...)  │    │              │    │  chunks)     │    │             │  │
│   └──────────┘    └──────────────┘    └──────────────┘    └──────┬──────┘  │
│                                                                   │         │
│                                                                   ▼         │
│                                                           ┌─────────────┐   │
│                                                           │  ChromaDB   │   │
│                                                           │  (Vector    │   │
│                                                           │   Store)    │   │
│                                                           └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                QUERY PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│   │  User    │───>│  Embedder    │───>│  Retriever   │───>│   Ollama    │  │
│   │  Query   │    │  (Vectorize  │    │  (Semantic   │    │   (LLM)     │  │
│   │          │    │   query)     │    │   search)    │    │             │  │
│   └──────────┘    └──────────────┘    └──────┬───────┘    └──────┬──────┘  │
│                                              │                    │         │
│                                              ▼                    ▼         │
│                                       ┌─────────────┐     ┌─────────────┐  │
│                                       │  ChromaDB   │     │  Response   │  │
│                                       │  (Top-K     │     │  (Answer +  │  │
│                                       │   results)  │     │   Sources)  │  │
│                                       └─────────────┘     └─────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Document Loader | PyPDF2, python-docx | Extract text from various file formats |
| Text Chunker | Custom recursive splitter | Split documents into overlapping chunks |
| Embedder | sentence-transformers (all-MiniLM-L6-v2) | Convert text to 384-dim vectors |
| Vector Store | ChromaDB | Store and search embeddings with cosine similarity |
| LLM | Ollama (llama3.2) | Generate answers from retrieved context |

## Project Structure

```
├── src/
│   ├── ingestion/      # Document loaders
│   ├── chunking/       # Text splitting
│   ├── embeddings/     # Sentence-transformers
│   ├── vectorstore/    # ChromaDB integration
│   ├── retrieval/      # Semantic search
│   ├── llm/            # Ollama client
│   ├── api/            # FastAPI endpoints
│   ├── config.py       # Settings
│   └── rag_pipeline.py # Main orchestrator
├── ui/
│   └── app.py          # Gradio interface
├── cli.py              # Command-line interface
└── requirements.txt
```

## License

MIT
