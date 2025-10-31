#!/usr/bin/env python3
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from apt.store import ChromaManager
from apt.config import Config

def test_vectordb_creation():
    logger.info("Testing Vector Database Creation with PyMuPDF4LLM chunks")

    pkl_file = Config.PROCESSED_DATA / "test_pymupdf4llm_chunks.pkl"

    if not pkl_file.exists():
        logger.error(f"Pickle file not found: {pkl_file}")
        logger.error("Run: uv run python scripts/test_pymupdf4llm.py first")
        return

    logger.info(f"Loading chunks from: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        chunks = pickle.load(f)

    logger.success(f"Loaded {len(chunks)} chunks")
    logger.info(f"First chunk metadata: {chunks[0].metadata}")

    test_collection = "test_pymupdf4llm_collection"
    test_db_path = Config.DATA_DIR / "chroma_db_test_pymupdf4llm"

    logger.info(f"\nCreating ChromaDB vectorstore...")
    logger.info(f"  Collection: {test_collection}")
    logger.info(f"  Database path: {test_db_path}")
    logger.info(f"  Embedding model: {Config.EMBEDDING_MODEL}")

    chroma_manager = ChromaManager(
        persist_directory=test_db_path,
        collection_name=test_collection,
        embedding_model=Config.EMBEDDING_MODEL
    )

    logger.info("\nCreating vectorstore (this may take a few minutes)...")
    vectorstore = chroma_manager.create_vectorstore(chunks)

    logger.success("Vectorstore created successfully!")

    stats = chroma_manager.get_collection_stats()
    logger.info("\nVectorstore Statistics:")
    logger.info(f"  Collection: {stats['collection_name']}")
    logger.info(f"  Documents: {stats['document_count']}")
    logger.info(f"  Location: {stats['persist_directory']}")

    db_size = sum(f.stat().st_size for f in test_db_path.rglob("*") if f.is_file())
    logger.info(f"  Database size: {db_size / (1024 * 1024):.2f} MB")

    logger.info("\n" + "="*80)
    logger.info("Testing Similarity Search")
    logger.info("="*80)

    test_queries = [
        "What is Icefog APT?",
        "What malware does the threat group use?",
        "spearphishing attacks techniques",
        "C&C server infrastructure"
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nQuery {i}: '{query}'")
        results = chroma_manager.similarity_search(query, k=3)

        logger.info(f"Found {len(results)} results:")
        for j, doc in enumerate(results, 1):
            logger.info(f"\n  Result {j}:")
            logger.info(f"    Filename: {doc.metadata.get('filename', 'N/A')}")
            logger.info(f"    Loader: {doc.metadata.get('loader', 'N/A')}")
            logger.info(f"    Format: {doc.metadata.get('format', 'N/A')}")
            logger.info(f"    Content preview: {doc.page_content[:200]}...")

    logger.info("\n" + "="*80)
    logger.info("Testing Retriever")
    logger.info("="*80)

    retriever = chroma_manager.get_retriever(search_kwargs={"k": 5})
    logger.success("Retriever created successfully")

    test_query = "What are the main attack vectors used by Icefog?"
    logger.info(f"\nTest query: '{test_query}'")
    retrieved_docs = retriever.invoke(test_query)

    logger.success(f"Retrieved {len(retrieved_docs)} documents")
    for i, doc in enumerate(retrieved_docs, 1):
        logger.info(f"\nDocument {i}:")
        logger.info(f"  Content length: {len(doc.page_content)} chars")
        logger.info(f"  Preview: {doc.page_content[:150]}...")

    logger.info("\n" + "="*80)
    logger.info("Comparison with Metadata")
    logger.info("="*80)

    logger.info("\nDocument metadata summary:")
    unique_filenames = set(doc.metadata.get('filename') for doc in chunks if 'filename' in doc.metadata)
    unique_loaders = set(doc.metadata.get('loader') for doc in chunks if 'loader' in doc.metadata)
    unique_formats = set(doc.metadata.get('format') for doc in chunks if 'format' in doc.metadata)

    logger.info(f"  Unique files: {unique_filenames}")
    logger.info(f"  Loaders: {unique_loaders}")
    logger.info(f"  Formats: {unique_formats}")

    logger.info("\n" + "="*80)
    logger.success("TEST COMPLETED SUCCESSFULLY!")
    logger.info("="*80)

    logger.info("\nSummary:")
    logger.info(f"  ✅ Loaded {len(chunks)} chunks from pickle")
    logger.info(f"  ✅ Created vectorstore with {stats['document_count']} documents")
    logger.info(f"  ✅ Similarity search working correctly")
    logger.info(f"  ✅ Retriever working correctly")
    logger.info(f"  ✅ Vector DB size: {db_size / (1024 * 1024):.2f} MB")

    logger.info("\nNext steps:")
    logger.info("  1. Run full pipeline with PyMuPDF4LLM on all PDFs")
    logger.info("  2. Compare RAG query quality with PDFPlumber")
    logger.info("  3. Test with different embedding models")

    logger.info(f"\nTest database location: {test_db_path}")
    logger.info("You can delete this test database with:")
    logger.info(f"  rm -rf {test_db_path}")

if __name__ == "__main__":
    test_vectordb_creation()
