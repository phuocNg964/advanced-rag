# Agentic RAG on My Notes

A RAG system for chatting with your PDF documents. Upload files, ask questions, and get answers with source citations.

Built with **LangGraph** (agentic workflow), **Weaviate** (vector database with local embeddings & reranking), **FastAPI** (API), and **Gemini** (LLM).

<!-- TODO: Replace with your own demo -->
<!-- ![Demo](docs/demo.gif) -->

<p align="center">
  <img src="docs/chat_demo.png" alt="Chat interface" width="700">
  <br>
  <em>Ask questions → get answers with inline citations</em>
</p>

<p align="center">
  <img src="docs/upload_demo.png" alt="Upload documents" width="700">
  <br>
  <em>Upload PDFs → background ingestion with status tracking</em>
</p>

---

## What It Does

1. **Upload PDFs** → extracts text, images, and tables → stores chunks in Weaviate
2. **Ask questions** → rewrites your query → hybrid search + reranking → generates answer with citations
3. **Multimodal** → images/tables are captioned by LLM and included as visual context during generation

## Document Ingestion Pipeline

<!-- TODO: Replace with your ingestion process screenshot or diagram -->
<p align="center">
  <img src="docs/ingestion_pipeline.png" alt="Ingestion Pipeline" width="700">
  <br>
  <em>Document ingestion pipeline — PDF parsing, image summarization, and chunking</em>
</p>

```
PDF Upload
    ↓
Partition (unstructured, hi_res strategy)
    • Extracts text, images, and tables from PDF
    • Saves images/tables as separate files (150 DPI)
    • OCR on individual blocks (not full-page)
    ↓
Filter & Clean
    • Removes headers, uncategorized text, tiny elements
    • Discards images < 10KB (logos, icons, etc.)
    ↓
Caption Attachment
    • Links nearby captions to their images/tables
    • Removes standalone caption elements to avoid duplicates
    ↓
Image Summarization (parallel, 5 workers)
    • Sends each image + its caption to Gemini vision
    • Generates information-dense text summary for vector search
    • Runs in ThreadPoolExecutor for speed on image-heavy PDFs
    ↓
Two-Stage Chunking
    • Stage 1: chunk_by_title (groups text under headings, max 10K chars)
    • Stage 2: RecursiveCharacterTextSplitter (1500 chars, no overlap)
    • Images/tables kept as single chunks with their summaries
    ↓
Store in Weaviate
    • Batch insert with content-based UUID (deduplication)
    • Metadata: source, page number, type, caption, image path
```

Each chunk is stored with its type (`Text`, `Image`, `Table`), so the generator knows when to include visual context during answer generation.

## How the Pipeline Works

<!-- TODO: Replace with your workflow diagram (e.g. LangGraph visualization or hand-drawn diagram) -->
<!-- ![RAG Workflow](docs/rag_workflow.png) -->

```
User Query
    ↓
Query Rewriter (Gemini Flash)
    • Resolves pronouns from chat history
    • Fixes typos, strips filler
    • Splits into 1–3 sub-queries if needed
    ↓
Hybrid Search (per sub-query)
    • BM25 + Vector search (α=0.5)
    • Cross-encoder reranking (top 25 → top 7)
    • Deduplicate across sub-queries
    ↓
Generator (Gemini Flash)
    • Synthesizes answer from retrieved docs
    • Includes images as base64 for vision context
    • Adds inline [1][2] citations
    ↓
Response with Source Citations
```

## Tech Stack

| Component | Technology |
|---|---|
| Agentic workflow | LangGraph (3-node graph: rewriter → retriever → generator) |
| Vector database | Weaviate (self-hosted via Docker) |
| Embeddings | `google-gemma-3-300m-embedding` (runs locally in Docker) |
| Reranker | `cross-encoder-ms-marco-MiniLM-L-6-v2` (runs locally in Docker) |
| LLM | Gemini 2.0 Flash (query rewriting) + Gemini 2.5 Flash Lite (generation) |
| PDF processing | Unstructured (text/image/table extraction) |
| API | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |

## Project Structure

```
├── src/
│   ├── rag_workflow.py       # LangGraph agent (query rewriter → retriever → generator)
│   ├── retriever.py          # Hybrid search + reranking
│   ├── ingest.py             # PDF processing and Weaviate ingestion
│   ├── api.py                # FastAPI endpoints
│   ├── collection_service.py # Weaviate collection CRUD
│   ├── config.py             # Settings (pydantic-settings)
│   ├── utils.py              # Helpers (base64 encoding, formatters)
│   └── logging_config.py     # Logging
├── static/                   # Chat UI (index.html, style.css, app.js)
├── notebooks/                # Experiments & evaluation (see notebooks/README.md)
├── docker-compose.yaml       # Weaviate + local embeddings + reranker
├── main.py                   # Entry point
├── requirements.txt
└── requirements-dev.txt      # Notebook & eval dependencies
```

## Getting Started

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- [Gemini API key](https://aistudio.google.com/apikey)

### Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/agentic-rag.git
cd agentic-rag

# 2. Environment
cp .env.example .env
# Edit .env → add your GEMINI_API_KEY

# 3. Start Weaviate (embeddings + reranker run locally)
docker compose up -d

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python main.py
# → http://localhost:8000
```

### Usage

1. Open `http://localhost:8000`
2. Create a collection in the sidebar
3. Upload PDF files (ingestion runs in background)
4. Start asking questions

## API

```
GET    /health                                    # Health check
GET    /collections                               # List collections
POST   /collections              {"name": "..."}  # Create collection
DELETE /collections/{name}                         # Delete collection

POST   /collections/{name}/documents  (file)      # Upload & ingest PDF
GET    /collections/{name}/documents              # List documents
DELETE /collections/{name}/documents/{filename}   # Delete document

GET    /jobs/{job_id}                              # Ingestion job status

POST   /collections/{name}/chat                   # Chat with documents
       {"message": "...", "session_id": "..."}
```

## Evaluation

Evaluated with [RAGAS](https://docs.ragas.io/) on a synthetic test set (21 queries) across 3 difficulty levels.

### Generation Quality

| Metric | Mean |
|---|---|
| Faithfulness | 0.92 |
| Answer Relevancy | 0.82 |

### Retrieval Quality

| Query Type | Samples | Recall | Precision |
|---|---|---|---|
| Single-hop specific | 7 | 0.86 | 0.71 |
| Multi-hop specific | 7 | 1.00 | 0.69 |
| Multi-hop abstract | 7 | 0.48 | 0.44 |

> Precision/recall use fuzzy (non-LLM) text matching against reference contexts.

**Key takeaways:**
- High faithfulness (0.92) — model rarely hallucinates beyond retrieved docs
- Multi-hop specific queries achieve perfect recall with decent precision
- Multi-hop abstract queries are the weakest — room for better query decomposition

Full evaluation pipeline: [`notebooks/synthetic_test_dataset.ipynb`](notebooks/synthetic_test_dataset.ipynb)

## Configuration

### Models (`src/rag_workflow.py`)

```python
'large_kwargs': {  # Generation
    'model': 'gemini-2.5-flash-lite',
    'temperature': 0.3,
},
'small_kwargs': {  # Query rewriting
    'model': 'gemini-2.0-flash',
    'temperature': 0.3,
},
```

### Retrieval (`src/retriever.py`)

```python
alpha = 0.5          # Hybrid weight (0=keyword, 1=vector)
top_k = 25           # Candidates before reranking
top_k_reranker = 7   # Results after reranking
```

### Environment Variables

| Variable | Required | Default |
|---|---|---|
| `GEMINI_API_KEY` | Yes | — |
| `WEAVIATE_HOST` | No | `localhost` |
| `WEAVIATE_HTTP_PORT` | No | `8080` |
| `WEAVIATE_GRPC_PORT` | No | `50051` |

## License

MIT
