#!/bin/bash
set -e

EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"

echo "=================================="
echo "APT RAG Pipeline - Full Execution"
echo "=================================="
echo ""
echo "Embedding Model: $EMBEDDING_MODEL"
echo "Start Time: $(date)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Working directory: $(pwd)"
echo ""

echo "Step 1/3: Downloading APT Reports..."
echo "--------------------------------------"
if [ -d "data/reports/aptnotes_pdfs" ] && [ "$(ls -A data/reports/aptnotes_pdfs)" ]; then
    echo "APT reports already downloaded ($(ls data/reports/aptnotes_pdfs | wc -l) files)"
else
    echo "Downloading reports from APTnotes..."
    tools/fetch
    echo "✓ Reports downloaded"
fi
echo ""

echo "Step 2/3: Extracting PDFs and Creating Chunks..."
echo "--------------------------------------"
if [ -f "data/processed/chunked_documents.pkl" ]; then
    echo "Chunked documents already exist, skipping extraction"
    echo "To re-extract, delete: data/processed/chunked_documents.pkl"
else
    tools/extract
    echo "✓ Extraction complete"
fi
echo ""

echo "Step 3/3: Creating Vector Embeddings..."
echo "--------------------------------------"
echo "This may take 20-40 minutes on GPU..."
tools/embed --model "$EMBEDDING_MODEL"
echo "✓ Embeddings created"
echo ""

echo "=================================="
echo "Pipeline Complete! 🎉"
echo "=================================="
echo "End Time: $(date)"
echo ""
echo "Output Files:"
echo "  - Vector DB: data/chroma_db/"
echo "  - Chunks: data/processed/chunked_documents.pkl"
echo "  - Logs: logs/"
echo ""
echo "Vector DB Size:"
du -sh data/chroma_db/ 2>/dev/null || echo "  Not yet created"
echo ""
echo "Test the system:"
echo "  tools/query 'Which APT group uses HrServ webshell?'"
echo ""
echo "To download results to local machine:"
echo "  bash deploy/download_results.sh"
echo ""
