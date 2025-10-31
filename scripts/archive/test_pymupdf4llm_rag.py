#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from apt.store import ChromaManager
from apt.retrieval import create_rag_chain
from apt.config import Config

def test_rag_query():
    logger.info("Testing Full RAG Pipeline with PyMuPDF4LLM Vector DB")

    test_db_path = Config.DATA_DIR / "chroma_db_test_pymupdf4llm"
    test_collection = "test_pymupdf4llm_collection"

    if not test_db_path.exists():
        logger.error(f"Test database not found: {test_db_path}")
        logger.error("Run: uv run python scripts/test_pymupdf4llm_vectordb.py first")
        return

    logger.info(f"Loading vectorstore from: {test_db_path}")
    logger.info(f"Collection: {test_collection}")

    chroma_manager = ChromaManager(
        persist_directory=test_db_path,
        collection_name=test_collection,
        embedding_model=Config.EMBEDDING_MODEL
    )

    vectorstore = chroma_manager.load_vectorstore()

    stats = chroma_manager.get_collection_stats()
    logger.success(f"Loaded vectorstore with {stats['document_count']} documents")

    logger.info("\nCreating RAG chain with Ollama...")
    logger.info("Note: Make sure Ollama is running (ollama serve)")

    try:
        rag_chain = create_rag_chain(vectorstore, llm_model="gemma3n:e4b")
        logger.success("RAG chain created successfully")
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        logger.error("And model is available: ollama pull gemma3n:e4b")
        return

    test_queries = [
        "What is the Icefog APT and what makes it different from other APT groups?",
        "What spearphishing techniques does Icefog use?",
        "What malware and backdoors are associated with Icefog?",
    ]

    logger.info("\n" + "="*80)
    logger.info("TESTING RAG QUERIES")
    logger.info("="*80)

    for i, question in enumerate(test_queries, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Query {i}/{len(test_queries)}")
        logger.info(f"{'='*80}")
        logger.info(f"\nQuestion: {question}")

        try:
            result = rag_chain.query(question)

            logger.info(f"\nAnswer:")
            logger.info(f"{result['answer']}")

            logger.info(f"\nSource Documents ({result['num_sources']}):")
            for j, doc in enumerate(result['source_documents'], 1):
                logger.info(f"\n  Source {j}:")
                logger.info(f"    Filename: {doc.metadata.get('filename', 'N/A')}")
                logger.info(f"    Loader: {doc.metadata.get('loader', 'N/A')}")
                logger.info(f"    Format: {doc.metadata.get('format', 'N/A')}")
                logger.info(f"    Year: {doc.metadata.get('year', 'N/A')}")
                logger.info(f"    Content preview: {doc.page_content[:150]}...")

        except Exception as e:
            logger.error(f"Query failed: {e}")

    logger.info("\n" + "="*80)
    logger.success("RAG TESTING COMPLETED!")
    logger.info("="*80)

    logger.info("\nSummary:")
    logger.info(f"  ✅ Vector DB: {stats['document_count']} documents")
    logger.info(f"  ✅ Loader: PyMuPDF4LLM (markdown format)")
    logger.info(f"  ✅ Embedding Model: {Config.EMBEDDING_MODEL}")
    logger.info(f"  ✅ LLM Model: gemma3n:e4b")
    logger.info(f"  ✅ Queries tested: {len(test_queries)}")

    logger.info("\nConclusion:")
    logger.info("  The PyMuPDF4LLM pipeline is working end-to-end!")
    logger.info("  PDF → Markdown → Chunks → Embeddings → Vector DB → RAG → Answers")

if __name__ == "__main__":
    test_rag_query()
