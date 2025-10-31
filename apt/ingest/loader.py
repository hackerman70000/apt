from pathlib import Path
from typing import List
from loguru import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
import pymupdf4llm
from apt.config import Config

class PDFLoader:
    def __init__(self, pdf_directory: Path, loader_type: str = None):
        self.pdf_directory = Path(pdf_directory)
        if not self.pdf_directory.exists():
            raise ValueError(f"PDF directory does not exist: {self.pdf_directory}")

        self.loader_type = loader_type or Config.PDF_LOADER
        logger.info(f"Using PDF loader: {self.loader_type}")

    def load_pdf(self, pdf_path: Path) -> List[Document]:
        try:
            if self.loader_type == "pymupdf4llm":
                md_text = pymupdf4llm.to_markdown(str(pdf_path))

                documents = [Document(
                    page_content=md_text,
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "loader": "pymupdf4llm",
                        "format": "markdown"
                    }
                )]

                if pdf_path.parent.name.isdigit():
                    documents[0].metadata["year"] = int(pdf_path.parent.name)

            else:
                loader = PDFPlumberLoader(str(pdf_path))
                documents = loader.load()

                if not documents:
                    logger.warning(f"No documents extracted from {pdf_path.name}")
                    return []

                for doc in documents:
                    doc.metadata["filename"] = pdf_path.name
                    doc.metadata["loader"] = "pdfplumber"
                    if pdf_path.parent.name.isdigit():
                        doc.metadata["year"] = int(pdf_path.parent.name)

            return documents

        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            return []

    def load_directory(self, max_files: int = None) -> List[Document]:
        pdf_files = list(self.pdf_directory.rglob("*.pdf"))

        if max_files:
            pdf_files = pdf_files[:max_files]

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        all_documents = []
        success_count = 0
        fail_count = 0

        for pdf_path in pdf_files:
            documents = self.load_pdf(pdf_path)
            if documents:
                all_documents.extend(documents)
                success_count += 1
            else:
                fail_count += 1

        logger.success(f"Loaded {success_count} PDFs successfully, {fail_count} failed")
        return all_documents

def load_pdfs(pdf_directory: Path = None, max_files: int = None, loader_type: str = None) -> List[Document]:
    if pdf_directory is None:
        pdf_directory = Config.REPORTS_DIR / "aptnotes_pdfs"

    if not pdf_directory.exists():
        logger.error(f"PDF directory not found: {pdf_directory}")
        return []

    loader = PDFLoader(pdf_directory, loader_type=loader_type)
    return loader.load_directory(max_files=max_files)
