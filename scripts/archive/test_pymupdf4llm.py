#!/usr/bin/env python3
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from apt.ingest.loader import load_pdfs
from apt.ingest.chunker import chunk_documents
from apt.config import Config

def test_pymupdf4llm_loader():
    logger.info("Testing PyMuPDF4LLM loader on single PDF")

    test_pdf = Config.REPORTS_DIR / "aptnotes_pdfs" / "2013" / "icefog.pdf"

    if not test_pdf.exists():
        logger.error(f"Test PDF not found: {test_pdf}")
        return

    logger.info(f"Loading PDF: {test_pdf.name}")
    logger.info(f"Using PyMuPDF4LLM loader (markdown format)")

    documents = load_pdfs(
        pdf_directory=test_pdf.parent,
        max_files=1,
        loader_type="pymupdf4llm"
    )

    if not documents:
        logger.error("No documents extracted!")
        return

    logger.success(f"Extracted {len(documents)} document(s)")

    for i, doc in enumerate(documents, 1):
        logger.info(f"\nDocument {i}:")
        logger.info(f"  Metadata: {doc.metadata}")
        logger.info(f"  Content length: {len(doc.page_content)} chars")
        logger.info(f"  Format: {doc.metadata.get('format', 'N/A')}")
        logger.info(f"  Loader: {doc.metadata.get('loader', 'N/A')}")
        logger.info(f"\nContent preview (first 500 chars):")
        logger.info(f"{doc.page_content[:500]}")

    logger.info("\nChunking document...")
    chunks = chunk_documents(documents)
    logger.success(f"Created {len(chunks)} chunks")

    logger.info(f"\nChunk preview (first 3 chunks):")
    for i, chunk in enumerate(chunks[:3], 1):
        logger.info(f"\nChunk {i}:")
        logger.info(f"  Length: {len(chunk.page_content)} chars")
        logger.info(f"  Metadata: {chunk.metadata}")
        logger.info(f"  Content preview: {chunk.page_content[:200]}...")

    output_file = Config.PROCESSED_DATA / "test_pymupdf4llm_chunks.pkl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(chunks, f)

    logger.success(f"\nSaved {len(chunks)} chunks to {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")

    logger.info("\nVerifying pickle file...")
    with open(output_file, 'rb') as f:
        loaded_chunks = pickle.load(f)

    logger.success(f"Successfully loaded {len(loaded_chunks)} chunks from pickle")

    markdown_output = Config.PROCESSED_DATA / "test_pymupdf4llm_full.md"
    with open(markdown_output, 'w', encoding='utf-8') as f:
        f.write(documents[0].page_content)

    logger.success(f"Saved full markdown to {markdown_output}")

    logger.info("\nTest completed successfully!")
    logger.info(f"  - Original document: 1")
    logger.info(f"  - Chunks created: {len(chunks)}")
    logger.info(f"  - Pickle file: {output_file}")
    logger.info(f"  - Markdown file: {markdown_output}")

if __name__ == "__main__":
    test_pymupdf4llm_loader()
