import pytest
from pathlib import Path
from langchain_core.documents import Document
from apt.store.chroma import ChromaManager

class TestChromaManager:
    def test_init(self, tmp_data_dir):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(
            persist_directory=persist_dir,
            collection_name="test_collection"
        )

        assert manager.persist_directory == persist_dir
        assert manager.collection_name == "test_collection"
        assert persist_dir.exists()

    def test_create_vectorstore(self, tmp_data_dir, sample_chunks):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)

        vectorstore = manager.create_vectorstore(sample_chunks)

        assert vectorstore is not None
        assert manager.vectorstore is not None

    def test_load_vectorstore(self, tmp_data_dir, sample_chunks):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)
        manager.create_vectorstore(sample_chunks)

        new_manager = ChromaManager(persist_directory=persist_dir)
        vectorstore = new_manager.load_vectorstore()

        assert vectorstore is not None
        assert new_manager.vectorstore is not None

    def test_add_documents(self, tmp_data_dir, sample_chunks):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)
        manager.create_vectorstore(sample_chunks[:1])

        new_doc = Document(
            page_content="Additional APT information",
            metadata={"filename": "new.pdf", "year": 2024}
        )

        manager.add_documents([new_doc])

    def test_add_documents_without_init_raises_error(self, tmp_data_dir):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)

        with pytest.raises(ValueError, match="Vectorstore not initialized"):
            manager.add_documents([Document(page_content="test")])

    def test_similarity_search(self, tmp_data_dir, sample_chunks):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)
        manager.create_vectorstore(sample_chunks)

        results = manager.similarity_search("APT28", k=2)

        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_similarity_search_with_filter(self, tmp_data_dir, sample_chunks):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)
        manager.create_vectorstore(sample_chunks)

        results = manager.similarity_search(
            "APT group",
            k=5,
            filter={"year": 2023}
        )

        assert isinstance(results, list)

    def test_similarity_search_without_init_raises_error(self, tmp_data_dir):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)

        with pytest.raises(ValueError, match="Vectorstore not initialized"):
            manager.similarity_search("test")

    def test_get_retriever(self, tmp_data_dir, sample_chunks):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)
        manager.create_vectorstore(sample_chunks)

        retriever = manager.get_retriever()

        assert retriever is not None

    def test_get_retriever_with_custom_kwargs(self, tmp_data_dir, sample_chunks):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)
        manager.create_vectorstore(sample_chunks)

        retriever = manager.get_retriever(search_kwargs={"k": 10})

        assert retriever is not None

    def test_get_retriever_without_init_raises_error(self, tmp_data_dir):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)

        with pytest.raises(ValueError, match="Vectorstore not initialized"):
            manager.get_retriever()

    def test_get_collection_stats(self, tmp_data_dir, sample_chunks):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)
        manager.create_vectorstore(sample_chunks)

        stats = manager.get_collection_stats()

        assert "collection_name" in stats
        assert "document_count" in stats
        assert "persist_directory" in stats
        assert stats["document_count"] > 0

    def test_get_collection_stats_without_init_raises_error(self, tmp_data_dir):
        persist_dir = tmp_data_dir / "test_chroma"
        manager = ChromaManager(persist_directory=persist_dir)

        with pytest.raises(ValueError, match="Vectorstore not initialized"):
            manager.get_collection_stats()
