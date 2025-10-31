# APT Threat Intelligence RAG System

Retrieval-Augmented Generation system for querying Advanced Persistent Threat intelligence reports using natural language.

## Overview

This system enables semantic search and question-answering over APT threat intelligence reports using:
- **Vector Database**: ChromaDB for document embeddings
- **Embedding Model**: Qwen/Qwen3-Embedding-8B (GPU-accelerated)
- **LLM**: Ollama (gemma3n:e4b)
- **Framework**: LangChain

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) installed and running
- HuggingFace token (for Qwen embedding model)

### Installation

```bash
uv sync

cp .env.example .env
```

Edit `.env` and add your HuggingFace token:
```bash
HF_TOKEN=your_huggingface_token_here
```

### Download APT Reports

```bash
tools/fetch
```

Downloads ~668 PDF reports (~1.2GB) to `data/reports/aptnotes_pdfs/`.

### Build Vector Database

```bash
tools/extract
tools/embed --model Qwen/Qwen3-Embedding-8B
```

Tools use `#!/usr/bin/env -S uv run` shebang, so dependencies are automatically loaded from `pyproject.toml`.

### Start Ollama

```bash
ollama serve
ollama pull gemma3n:e4b
```

### Query the System

```bash
tools/query "Which APT group uses HrServ webshell?"
tools/query --model llama3.2:latest "What are the TTPs of APT28?"
```

## Project Structure

```
apt/
├── apt/                    # Core library (importable package)
│   ├── ingest/            # PDF loading, chunking, metadata
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── metadata.py
│   ├── store/             # Vector database management
│   │   └── chroma.py
│   ├── retrieval/         # RAG chain implementation
│   │   └── chain.py
│   ├── config.py          # Configuration
│   └── __init__.py
├── tools/                 # CLI utilities (executable)
│   ├── extract            # PDF extraction & chunking
│   ├── embed              # Create embeddings
│   ├── query              # Query RAG system
│   └── fetch              # Download reports
├── deploy/                # Cloud GPU deployment
│   ├── startup.sh         # Cloud setup script
│   ├── run_pipeline.sh    # Full pipeline execution
│   └── download_results.sh
├── docs/                  # Documentation
│   ├── DEPLOYMENT.md      # Cloud deployment guide
│   └── EMBEDDING_MODEL_CHOICE.md
├── experiments/           # Research experiments
├── data/                  # All data
│   ├── chroma_db/         # Vector database
│   ├── processed/         # Chunked documents
│   ├── logs/              # Execution logs
│   └── reports/           # APT reports
├── pyproject.toml         # Dependencies
├── .gitignore
├── .env.example
└── README.md
```

## Usage

### CLI Tools

All tools are executable scripts using `uv run` shebang. You can run them directly:

```bash
tools/fetch                                    # Download APT reports
tools/extract                                 # Extract & chunk PDFs
tools/extract --max-files 100                 # Process first 100 PDFs
tools/embed --model Qwen/Qwen3-Embedding-8B   # Create embeddings
tools/query "Your question here"              # Query the system
tools/query --model llama3.2 "Question"       # Use different LLM
```

Or explicitly with `uv run`:

```bash
uv run tools/fetch
uv run tools/extract
uv run tools/embed --model Qwen/Qwen3-Embedding-8B
uv run tools/query "Your question"
```

### As a Library

```python
from apt.ingest import load_pdfs, chunk_documents
from apt.store import ChromaManager
from apt.retrieval import create_rag_chain

documents = load_pdfs(max_files=10)
chunks = chunk_documents(documents)

chroma = ChromaManager()
chroma.create_vectorstore(chunks)

vectorstore = chroma.load_vectorstore()
rag_chain = create_rag_chain(vectorstore)

result = rag_chain.query("Which APT group uses HrServ webshell?")
print(result["answer"])
```

## Cloud GPU Deployment

Deploy on cloud GPU instances (Lambda Labs, Vast.ai, RunPod) for faster embedding creation.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed guide.

### Quick Deploy

1. Push to GitHub
2. Create GPU instance with startup script:
   ```bash
   export REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO.git"
   export HF_TOKEN="hf_xxxxx"
   curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/deploy/startup.sh | bash
   ```
3. SSH and run pipeline:
   ```bash
   cd /workspace/apt-rag
   bash deploy/run_pipeline.sh
   ```
4. Download results:
   ```bash
   bash deploy/download_results.sh
   scp user@host:/workspace/apt-rag/rag_outputs_*.tar.gz .
   ```

## Features

- **Semantic Search**: Natural language queries over APT reports
- **RAG Question Answering**: Detailed answers with source attribution
- **Metadata Enrichment**: Automatic extraction of APT groups and MITRE techniques
- **GPU Acceleration**: Auto-detection and use of CUDA when available
- **Modern CLI**: Typer-based tools with rich output
- **Cloud Ready**: Deployment scripts for cloud GPU providers
- **Flexible Models**: Support for multiple embedding and LLM models

## Example Queries

```bash
tools/query "Which APT group uses HrServ webshell?"
tools/query "What are the TTPs of APT28?"
tools/query "Show me spearphishing campaigns from 2023"
tools/query "What malware does Lazarus Group use?"
```

## Configuration

Edit `apt/config.py` or set environment variable:

```python
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
COLLECTION_NAME = "apt_reports"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5
```

## Data Sources

### APTnotes Repository
- **Source**: https://github.com/kbandla/APTnotes
- **Content**: 668 APT threat intelligence reports
- **Coverage**: 2006-2024
- **Format**: PDF reports from major threat intelligence vendors

Reports include APT group attributions, malware analysis, campaign details, IOCs, and MITRE ATT&CK techniques.

## Technical Details

### Pipeline Stages

1. **PDF Extraction** (`tools/extract`)
   - Loads PDFs with PDFPlumber
   - Chunks text (1000 chars, 200 overlap)
   - Enriches with metadata
   - Output: `data/processed/chunked_documents.pkl`

2. **Embedding Creation** (`tools/embed`)
   - Creates embeddings with Qwen model
   - Stores in ChromaDB
   - GPU-accelerated when available
   - Output: `data/chroma_db/`

3. **Query** (`tools/query`)
   - Semantic search via embeddings
   - LLM-powered answer generation
   - Source attribution with metadata

### GPU Support

Automatic CUDA detection and usage:
- CPU: ~40-60 minutes for embeddings
- GPU: ~10-20 minutes for embeddings

## Development

```bash
uv sync

tools/extract
tools/embed --model YOUR_MODEL
tools/query "Your question"
```

## License

MIT License

## Citation

If you use this work, please cite:
```
APT Threat Intelligence RAG System, 2025
```
