import pytest
import pickle
import shutil
from pathlib import Path
from langchain_core.documents import Document
from apt.ingest.loader import load_pdfs
from apt.ingest.chunker import chunk_documents
from apt.store import ChromaManager
from apt.retrieval import create_rag_chain
from apt.config import Config

@pytest.fixture(scope="class")
def test_pdf_path():
    pdf_path = Config.REPORTS_DIR / "aptnotes_pdfs" / "2013" / "icefog.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    return pdf_path

@pytest.fixture(scope="class")
def test_output_dir(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("pymupdf4llm_test")
    yield output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)

@pytest.fixture(scope="class")
def test_db_path(test_output_dir):
    db_path = test_output_dir / "chroma_db"
    yield db_path
    if db_path.exists():
        shutil.rmtree(db_path)

@pytest.fixture(scope="class")
def extracted_documents(test_pdf_path):
    documents = load_pdfs(
        pdf_directory=test_pdf_path.parent,
        max_files=1,
        loader_type="pymupdf4llm"
    )
    return documents

@pytest.fixture(scope="class")
def chunked_documents(extracted_documents):
    chunks = chunk_documents(extracted_documents)
    return chunks

@pytest.fixture(scope="class")
def pickle_file(chunked_documents, test_output_dir):
    pkl_file = test_output_dir / "test_chunks.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(chunked_documents, f)
    yield pkl_file
    if pkl_file.exists():
        pkl_file.unlink()

@pytest.fixture(scope="class")
def vectorstore(chunked_documents, test_db_path):
    chroma_manager = ChromaManager(
        persist_directory=test_db_path,
        collection_name="test_pymupdf4llm",
        embedding_model=Config.EMBEDDING_MODEL
    )
    vectorstore = chroma_manager.create_vectorstore(chunked_documents)
    yield vectorstore

@pytest.fixture(scope="class")
def chroma_manager(vectorstore, test_db_path):
    manager = ChromaManager(
        persist_directory=test_db_path,
        collection_name="test_pymupdf4llm",
        embedding_model=Config.EMBEDDING_MODEL
    )
    manager.vectorstore = vectorstore
    return manager

@pytest.mark.integration
class TestPyMuPDF4LLMPipeline:

    def test_01_pdf_extraction(self, extracted_documents, test_pdf_path):
        assert extracted_documents is not None
        assert len(extracted_documents) > 0

        doc = extracted_documents[0]
        assert doc.page_content is not None
        assert len(doc.page_content) > 0

        assert "filename" in doc.metadata
        assert doc.metadata["filename"] == test_pdf_path.name
        assert doc.metadata["loader"] == "pymupdf4llm"
        assert doc.metadata["format"] == "markdown"
        assert "year" in doc.metadata

        assert "# " in doc.page_content or "## " in doc.page_content

    def test_02_chunking(self, chunked_documents, extracted_documents):
        assert chunked_documents is not None
        assert len(chunked_documents) > 0
        assert len(chunked_documents) > len(extracted_documents)

        first_chunk = chunked_documents[0]
        assert isinstance(first_chunk, Document)
        assert len(first_chunk.page_content) > 0
        assert first_chunk.metadata["loader"] == "pymupdf4llm"
        assert first_chunk.metadata["format"] == "markdown"

    def test_03_pickle_serialization(self, pickle_file, chunked_documents):
        assert pickle_file.exists()
        assert pickle_file.stat().st_size > 0

        with open(pickle_file, 'rb') as f:
            loaded_chunks = pickle.load(f)

        assert len(loaded_chunks) == len(chunked_documents)
        assert isinstance(loaded_chunks[0], Document)
        assert loaded_chunks[0].metadata["loader"] == "pymupdf4llm"

    def test_04_vectorstore_creation(self, vectorstore, chunked_documents):
        assert vectorstore is not None

        collection = vectorstore._collection
        count = collection.count()
        assert count == len(chunked_documents)

    def test_05_similarity_search(self, chroma_manager):
        test_queries = [
            "What is Icefog APT?",
            "spearphishing techniques",
            "malware"
        ]

        for query in test_queries:
            results = chroma_manager.similarity_search(query, k=3)
            assert len(results) > 0
            assert len(results) <= 3

            for doc in results:
                assert isinstance(doc, Document)
                assert "loader" in doc.metadata
                assert doc.metadata["loader"] == "pymupdf4llm"

    def test_06_retriever(self, chroma_manager):
        retriever = chroma_manager.get_retriever(search_kwargs={"k": 5})
        assert retriever is not None

        test_query = "What are the main attack vectors?"
        results = retriever.invoke(test_query)

        assert len(results) > 0
        assert len(results) <= 5

        for doc in results:
            assert isinstance(doc, Document)
            assert len(doc.page_content) > 0

    def test_07_collection_stats(self, chroma_manager, chunked_documents):
        stats = chroma_manager.get_collection_stats()

        assert "collection_name" in stats
        assert stats["collection_name"] == "test_pymupdf4llm"
        assert "document_count" in stats
        assert stats["document_count"] == len(chunked_documents)
        assert "persist_directory" in stats

    def test_08_metadata_preservation(self, chunked_documents):
        for chunk in chunked_documents:
            assert "filename" in chunk.metadata
            assert "loader" in chunk.metadata
            assert chunk.metadata["loader"] == "pymupdf4llm"
            assert "format" in chunk.metadata
            assert chunk.metadata["format"] == "markdown"

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_ollama
class TestPyMuPDF4LLMRAGQueries:

    @pytest.fixture(scope="class")
    def rag_db_path(self, test_output_dir):
        db_path = test_output_dir / "chroma_db_rag"
        yield db_path
        if db_path.exists():
            shutil.rmtree(db_path)

    @pytest.fixture(scope="class")
    def rag_vectorstore(self, rag_db_path, chunked_documents):
        manager = ChromaManager(
            persist_directory=rag_db_path,
            collection_name="test_pymupdf4llm_rag",
            embedding_model=Config.EMBEDDING_MODEL
        )
        vectorstore = manager.create_vectorstore(chunked_documents)
        yield vectorstore

    @pytest.fixture(scope="class")
    def rag_chain(self, rag_vectorstore):
        pytest.importorskip("ollama", reason="Ollama not installed")

        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code != 200:
                pytest.skip("Ollama not running. Start with: ollama serve")
        except Exception:
            pytest.skip("Ollama not running. Start with: ollama serve")

        try:
            chain = create_rag_chain(rag_vectorstore, llm_model="gemma3n:e4b")
            return chain
        except Exception as e:
            pytest.skip(f"Failed to create RAG chain: {e}")

    def test_rag_query_basic(self, rag_chain):
        result = rag_chain.query("What is Icefog?")

        assert "question" in result
        assert "answer" in result
        assert "source_documents" in result
        assert "num_sources" in result

        assert len(result["answer"]) > 0
        assert len(result["source_documents"]) > 0

    def test_rag_query_spearphishing(self, rag_chain):
        result = rag_chain.query("What spearphishing techniques does Icefog use?")

        assert len(result["answer"]) > 0
        assert result["num_sources"] > 0

        for doc in result["source_documents"]:
            assert doc.metadata["loader"] == "pymupdf4llm"

    def test_rag_source_attribution(self, rag_chain):
        result = rag_chain.query("What malware does Icefog use?")

        assert result["num_sources"] > 0

        for doc in result["source_documents"]:
            assert "filename" in doc.metadata
            assert "loader" in doc.metadata
            assert doc.metadata["loader"] == "pymupdf4llm"
