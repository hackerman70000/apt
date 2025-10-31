from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger
from apt.config import Config
from apt.ingest.metadata import extract_apt_mentions, extract_technique_mentions

class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Chunking {len(documents)} documents")

        chunks = self.text_splitter.split_documents(documents)

        logger.success(f"Created {len(chunks)} chunks from {len(documents)} documents")

        return chunks

    def enrich_metadata(self, chunks: List[Document]) -> List[Document]:
        logger.info("Enriching chunks with custom metadata")

        for chunk in chunks:
            apt_groups = extract_apt_mentions(chunk.page_content)
            if apt_groups:
                chunk.metadata["apt_groups_mentioned"] = ", ".join(sorted(apt_groups))

            techniques = extract_technique_mentions(chunk.page_content)
            if techniques:
                chunk.metadata["techniques_mentioned"] = ", ".join(sorted(techniques))

        return chunks

def chunk_documents(documents: List[Document], chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    chunker = DocumentChunker(
        chunk_size=chunk_size or Config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or Config.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(documents)
    return chunker.enrich_metadata(chunks)
