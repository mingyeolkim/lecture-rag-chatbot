# Lecture RAG Chatbot

A Korean lecture-material question-answering prototype. It extracts text from PDFs, chunks it with token overlap, stores dense embeddings in ChromaDB, combines dense retrieval with BM25, reranks candidates, and generates a short Korean answer with a locally loaded LLaMA model. A command-line chat interface and a small evaluation script are included.

This repository publishes the **source code and configuration**, not the lecture PDF, its extracted text, vector index, local model weights, or the original evaluation questions. Those inputs were not cleared for public redistribution. The code originated in `/workspace/llm`; the public package is a curated copy rather than a dump of the original workspace or its ZIP archive.

## Pipeline

`PDF → page text → 300–700-token overlapping chunks → BGE-M3 dense embeddings + Chroma index → dense/BM25 hybrid search → BGE reranker → local LLaMA generation → Korean answer post-processing`

The default model IDs and retrieval parameters are in [`configs/config.yaml`](configs/config.yaml). `rag.py` loads the embedding model, reranker, Chroma collection, chunks, and LLaMA **at import time**. Expect a substantial model download and memory requirement; this is not a lightweight CPU-only demo. Model access and license terms are governed by their respective providers.

## Set up

Use Python 3.10 and install a PyTorch build appropriate for your system, then install [`requirements.txt`](requirements.txt). Place PDFs you are authorized to use under `data/raw_pdfs/`. From the repository root:

```bash
python ingest.py
python cli_chat.py
```

`ingest.py` builds `data/chunks.parquet` and `index/chroma/`. **It deletes and recreates the Chroma collection named `lectures`** each time, so back up any index you want to preserve before rerunning it. `cli_chat.py` accepts questions, `exit`/`quit`, and `metric` to evaluate the configured QA file. These commands have not been end-to-end rerun as part of the GitHub packaging because the PDF, index and model assets are excluded.

To evaluate on your own authorized questions, create `data/qa.csv` using [`data/qa.example.csv`](data/qa.example.csv) as the column-format example and run `python eval.py`. The example row is **synthetic** and is not an experimental result.

## Evaluation caveats

The original `eval.py` reports exact match, a simple *set-of-words* overlap F1, citation-page hit, a heuristic hallucination flag, and latency. These are exploratory proxies, **not** validated factuality or hallucination metrics. In particular, the generation prompt asks the model not to include source tags while the source-hit heuristic looks for `p.N` tags in the raw answer; interpret that number cautiously. No benchmark scores are claimed in this repository.

## Files

- `ingest.py`: PDF extraction, chunking, dense index creation.
- `rag.py`: retrieval, reranking, prompting, generation, answer cleanup.
- `utils.py`: text cleaning, token chunking, guardrail prompt.
- `cli_chat.py`: interactive CLI.
- `eval.py`: exploratory evaluation.
- `configs/config.yaml`: data paths and model/retrieval parameters.
- `tests/`: lightweight code-level checks that do not load the large models.

No source-code redistribution license has been selected yet; making a GitHub repository public does not itself grant reuse rights.
