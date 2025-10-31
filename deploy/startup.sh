#!/bin/bash
set -e

echo "=================================="
echo "APT RAG System - Cloud GPU Setup"
echo "=================================="
echo ""

REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/YOUR_REPO.git}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/apt-rag}"
HF_TOKEN="${HF_TOKEN:-}"

echo "Step 1: Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    git \
    curl \
    poppler-utils \
    tesseract-ocr \
    wget \
    > /dev/null 2>&1

echo "System dependencies installed"
echo ""

echo "Step 2: Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
    echo "Ollama installed"
else
    echo "Ollama already installed"
fi

echo "Starting Ollama service..."
nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama service started (PID: $OLLAMA_PID)"

sleep 5

echo "Pulling gemma3n:e4b model..."
ollama pull gemma3n:e4b
echo "Model ready"
echo ""

echo "Step 3: Cloning repository..."
if [ -d "$WORKSPACE_DIR" ]; then
    echo "Directory $WORKSPACE_DIR already exists, skipping clone"
else
    git clone "$REPO_URL" "$WORKSPACE_DIR"
    echo "Repository cloned to $WORKSPACE_DIR"
fi
cd "$WORKSPACE_DIR"
echo ""

echo "Step 4: Setting up Python environment..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "uv installed"
else
    echo "uv already installed"
fi

echo "Installing Python dependencies..."
uv sync
echo "Python environment ready"
echo ""

echo "Step 5: Configuring environment..."
cat > .env << EOF
HF_TOKEN=$HF_TOKEN
LANGSMITH_TRACING=false
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
PDF_LOADER=pymupdf4llm
EOF
echo "Environment variables configured"
echo ""

echo "========================================"
echo "Setup Complete"
echo "========================================"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU detected"
echo ""
echo "Ollama Status:"
ollama list
echo ""
echo "Next Steps:"
echo "1. SSH into this instance"
echo "2. cd $WORKSPACE_DIR"
echo "3. Run the pipeline:"
echo "   bash deploy/run_pipeline.sh"
echo ""
echo "Or run steps manually:"
echo "   tools/fetch"
echo "   tools/extract"
echo "   tools/embed --model Qwen/Qwen3-Embedding-8B"
echo ""
echo "After testing RAG queries, download results:"
echo "   bash deploy/download_results.sh"
echo ""
echo "========================================"
