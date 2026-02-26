# Notebooks

This directory contains Jupyter notebooks used for development, testing, and evaluation.

| Notebook | Purpose |
|---|---|
| **Agentic_Researcher.ipynb** | End-to-end demo of the Agentic RAG pipeline — query rewriting, retrieval, and generation |
| **ingest.ipynb** | Step-by-step test of the `IngestService`: loading, chunking, and storing documents in Weaviate |
| **test_ingest_service.ipynb** | Detailed unit-level testing of individual `IngestService` methods (connection, CRUD, retrieval) |
| **new_chunking_strategy.ipynb** | Experiments with improved chunking: image/table captioning, smaller chunk sizes, runtime optimisation |
| **synthetic_test_dataset.ipynb** | Generates a synthetic Q&A dataset and runs RAGAS evaluation (faithfulness, relevancy, recall) |

## Supporting Files

| File/Directory | Purpose |
|---|---|
| **utils.py** | Shared helper functions used across notebooks |
| **data/** | CSV outputs from evaluation runs (retrieval & generation metrics) |
