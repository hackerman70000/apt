from pathlib import Path
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from loguru import logger
from apt.config import Config

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class ChromaManager:
    def __init__(
        self,
        persist_directory: Path = Config.CHROMA_DB,
        collection_name: str = Config.COLLECTION_NAME,
        embedding_model: str = Config.EMBEDDING_MODEL,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(f"Initializing embeddings with model: {embedding_model}")

        device = "cpu"
        if HAS_TORCH and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for embeddings")

        model_kwargs = {"device": device}
        if Config.HF_TOKEN:
            model_kwargs["token"] = Config.HF_TOKEN

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
        )

        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document], batch_size: int = 100) -> Chroma:
        logger.info(f"Creating Chroma vectorstore with {len(documents)} documents")

        logger.info("Filtering complex metadata from documents")
        filtered_documents = filter_complex_metadata(documents)
        logger.success(f"Filtered {len(filtered_documents)} documents")

        total_docs = len(filtered_documents)

        # Process in batches with progress logging
        if total_docs > batch_size:
            logger.info(f"Processing documents in batches of {batch_size}")

            # Create vectorstore with first batch
            first_batch = filtered_documents[:batch_size]
            logger.info(f"Creating vectorstore with first batch ({batch_size} docs)")
            logger.info("Computing embeddings... (this may take a while)")

            self.vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
            )
            logger.success(f"First batch complete: {batch_size}/{total_docs}")

            # Add remaining batches
            for i in range(batch_size, total_docs, batch_size):
                batch = filtered_documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                progress_pct = (i / total_docs) * 100

                logger.info(f"Batch {batch_num}: Processing {i}-{min(i+batch_size, total_docs)}/{total_docs} ({progress_pct:.1f}%)")
                self.vectorstore.add_documents(batch)
                logger.success(f"Batch {batch_num} complete")
        else:
            logger.info(f"Processing all {total_docs} documents at once")
            logger.info("Computing embeddings... (this may take a while)")

            self.vectorstore = Chroma.from_documents(
                documents=filtered_documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
            )

        logger.success(f"Vectorstore created and persisted to {self.persist_directory}")
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        logger.info(f"Loading existing Chroma vectorstore from {self.persist_directory}")

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

        logger.success("Vectorstore loaded successfully")
        return self.vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")

        logger.info(f"Adding {len(documents)} documents to vectorstore")
        self.vectorstore.add_documents(documents)
        logger.success(f"Added {len(documents)} documents")

    def similarity_search(
        self,
        query: str,
        k: int = Config.RETRIEVAL_K,
        filter: Optional[dict] = None
    ) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")

        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )

        return results

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")

        if search_kwargs is None:
            search_kwargs = {"k": Config.RETRIEVAL_K}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def get_collection_stats(self) -> dict:
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")

        collection = self.vectorstore._collection
        count = collection.count()

        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": str(self.persist_directory),
        }
