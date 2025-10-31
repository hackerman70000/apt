#!/bin/bash
set -e

echo "=================================="
echo "Packaging Results for Download"
echo "=================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

OUTPUT_FILE="rag_outputs_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "Creating archive: $OUTPUT_FILE"
echo ""

tar -czf "$OUTPUT_FILE" \
    data/chroma_db/ \
    data/processed/chunked_documents.pkl \
    data/logs/ \
    2>/dev/null || true

ARCHIVE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo "Archive created successfully"
echo ""
echo "Archive: $OUTPUT_FILE"
echo "Size: $ARCHIVE_SIZE"
echo ""
echo "=================================="
echo "Download Instructions"
echo "=================================="
echo ""
echo "From your LOCAL machine, run:"
echo ""
echo "  scp USER@HOST:$(pwd)/$OUTPUT_FILE ."
echo ""
echo "Or if using cloud provider CLI:"
echo ""
echo "Lambda Labs:"
echo "  ssh USER@HOST 'cat $(pwd)/$OUTPUT_FILE' > $OUTPUT_FILE"
echo ""
echo "Then extract on your local machine:"
echo "  tar -xzf $OUTPUT_FILE -C /path/to/your/apt/project/"
echo ""
echo "This will restore:"
echo "  - data/chroma_db/          (vector database)"
echo "  - data/processed/          (chunked documents)"
echo "  - data/logs/               (execution logs)"
echo ""
