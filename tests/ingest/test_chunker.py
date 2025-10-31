import pytest
from langchain_core.documents import Document
from apt.ingest.chunker import DocumentChunker, chunk_documents

class TestDocumentChunker:
    def test_chunk_documents(self, sample_documents):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_documents(sample_documents)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_chunk_size_respected(self):
        doc = Document(page_content="A" * 500, metadata={"test": "value"})
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk_documents([doc])

        for chunk in chunks:
            assert len(chunk.page_content) <= 100

    def test_metadata_preserved(self, sample_documents):
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(sample_documents)

        for chunk in chunks:
            assert "filename" in chunk.metadata

    def test_enrich_metadata(self):
        doc = Document(
            page_content="APT28 used T1566.001 technique",
            metadata={"filename": "test.pdf"}
        )
        chunker = DocumentChunker()
        enriched = chunker.enrich_metadata([doc])

        assert enriched[0].metadata["apt_groups_mentioned"] == "APT28"
        assert "T1566.001" in enriched[0].metadata["techniques_mentioned"]

    def test_chunk_documents_function(self, sample_documents):
        chunks = chunk_documents(sample_documents)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_custom_chunk_size(self):
        doc = Document(page_content="A" * 1000, metadata={})
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 200
