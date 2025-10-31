import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from apt.retrieval.chain import RAGChain, create_rag_chain

class TestRAGChain:
    @pytest.fixture
    def mock_retriever(self):
        def retriever_func(query):
            return [
                Document(
                    page_content="APT28 uses spearphishing T1566.001",
                    metadata={"filename": "apt28.pdf", "year": 2023}
                ),
                Document(
                    page_content="The group targets government entities",
                    metadata={"filename": "apt28.pdf", "year": 2023}
                )
            ]
        return RunnableLambda(retriever_func)

    @pytest.fixture
    def mock_llm(self, mocker):
        def llm_func(prompt):
            return "APT28 is a Russian threat group."

        mock = mocker.patch("apt.retrieval.chain.OllamaLLM")
        mock_runnable = RunnableLambda(llm_func)
        mock.return_value = mock_runnable
        return mock_runnable

    def test_init(self, mock_retriever, mock_llm):
        chain = RAGChain(mock_retriever, llm_model="gemma3n:e4b")

        assert chain.retriever is not None
        assert chain.llm is not None
        assert chain.prompt is not None
        assert chain.chain is not None

    def test_format_docs(self, mock_retriever, mock_llm):
        chain = RAGChain(mock_retriever)

        docs = [
            Document(
                page_content="Test content " * 100,
                metadata={"filename": "test.pdf", "year": 2023}
            ),
            Document(
                page_content="Another content",
                metadata={"filename": "test2.pdf", "year": 2024}
            )
        ]

        formatted = chain._format_docs(docs)

        assert "[Report 1]" in formatted
        assert "[Report 2]" in formatted
        assert "test.pdf" in formatted
        assert "test2.pdf" in formatted
        assert "Year: 2023" in formatted
        assert "Year: 2024" in formatted

    def test_format_docs_truncates_content(self, mock_retriever, mock_llm):
        chain = RAGChain(mock_retriever)

        long_content = "A" * 1000
        docs = [
            Document(
                page_content=long_content,
                metadata={"filename": "test.pdf", "year": 2023}
            )
        ]

        formatted = chain._format_docs(docs)

        assert len(formatted) < len(long_content)
        assert "..." in formatted

    def test_format_docs_handles_missing_metadata(self, mock_retriever, mock_llm):
        chain = RAGChain(mock_retriever)

        docs = [
            Document(page_content="Content without metadata", metadata={})
        ]

        formatted = chain._format_docs(docs)

        assert "Unknown" in formatted
        assert "N/A" in formatted

    def test_query(self, mock_retriever, mock_llm):
        chain = RAGChain(mock_retriever)

        result = chain.query("What are APT28's tactics?")

        assert "question" in result
        assert "answer" in result
        assert "source_documents" in result
        assert "num_sources" in result
        assert result["question"] == "What are APT28's tactics?"
        assert len(result["source_documents"]) == 2

    def test_query_with_filter(self, mock_llm):
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [
            Document(
                page_content="Filtered result",
                metadata={"filename": "apt28.pdf", "year": 2023}
            )
        ]

        def retriever_func(query):
            return [
                Document(
                    page_content="Default result",
                    metadata={"filename": "apt28.pdf", "year": 2023}
                )
            ]

        mock_retriever = RunnableLambda(retriever_func)
        mock_retriever.vectorstore = mock_vectorstore

        chain = RAGChain(mock_retriever)

        result = chain.query_with_filter(
            "APT28 tactics",
            metadata_filter={"year": 2023}
        )

        assert "question" in result
        assert "answer" in result
        assert "source_documents" in result
        assert "filter" in result
        assert result["filter"] == {"year": 2023}

class TestCreateRAGChain:
    def test_create_rag_chain(self, mocker):
        mock_vectorstore = Mock()

        def retriever_func(query):
            return [
                Document(
                    page_content="Test content",
                    metadata={"filename": "test.pdf", "year": 2023}
                )
            ]

        mock_retriever = RunnableLambda(retriever_func)
        mock_vectorstore.as_retriever.return_value = mock_retriever

        mocker.patch("apt.retrieval.chain.OllamaLLM")

        chain = create_rag_chain(mock_vectorstore, llm_model="gemma3n:e4b")

        assert isinstance(chain, RAGChain)
        mock_vectorstore.as_retriever.assert_called_once()

    def test_create_rag_chain_default_model(self, mocker):
        mock_vectorstore = Mock()

        def retriever_func(query):
            return [
                Document(
                    page_content="Test content",
                    metadata={"filename": "test.pdf", "year": 2023}
                )
            ]

        mock_retriever = RunnableLambda(retriever_func)
        mock_vectorstore.as_retriever.return_value = mock_retriever

        mock_ollama = mocker.patch("apt.retrieval.chain.OllamaLLM")

        create_rag_chain(mock_vectorstore)

        mock_ollama.assert_called_once()
