#!/usr/bin/env -S uv run
from pathlib import Path
from typing import Dict, List
import time
from loguru import logger

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not available")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logger.warning("pdfplumber not available")

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    logger.warning("pypdf not available")

from langchain_community.document_loaders import (
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFLoader,
)

class PDFExtractorComparison:
    def __init__(self, pdf_path: Path, output_dir: Path):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_name = pdf_path.stem

    def extract_with_pymupdf_raw(self) -> Dict:
        if not HAS_PYMUPDF:
            return {"error": "PyMuPDF not installed"}

        start_time = time.time()
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            doc.close()

            elapsed = time.time() - start_time

            output_file = self.output_dir / f"{self.pdf_name}_pymupdf_raw.txt"
            output_file.write_text(text, encoding="utf-8")

            return {
                "method": "PyMuPDF (raw)",
                "time": elapsed,
                "text_length": len(text),
                "output_file": str(output_file),
                "success": True
            }
        except Exception as e:
            return {
                "method": "PyMuPDF (raw)",
                "error": str(e),
                "success": False
            }

    def extract_with_pdfplumber_raw(self) -> Dict:
        if not HAS_PDFPLUMBER:
            return {"error": "pdfplumber not installed"}

        start_time = time.time()
        try:
            text = ""
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text += f"\n--- Page {page_num + 1} ---\n"
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

            elapsed = time.time() - start_time

            output_file = self.output_dir / f"{self.pdf_name}_pdfplumber_raw.txt"
            output_file.write_text(text, encoding="utf-8")

            return {
                "method": "PDFPlumber (raw)",
                "time": elapsed,
                "text_length": len(text),
                "output_file": str(output_file),
                "success": True
            }
        except Exception as e:
            return {
                "method": "PDFPlumber (raw)",
                "error": str(e),
                "success": False
            }

    def extract_with_pypdf_raw(self) -> Dict:
        if not HAS_PYPDF:
            return {"error": "pypdf not installed"}

        start_time = time.time()
        try:
            reader = PdfReader(self.pdf_path)
            text = ""
            for page_num, page in enumerate(reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()

            elapsed = time.time() - start_time

            output_file = self.output_dir / f"{self.pdf_name}_pypdf_raw.txt"
            output_file.write_text(text, encoding="utf-8")

            return {
                "method": "PyPDF (raw)",
                "time": elapsed,
                "text_length": len(text),
                "output_file": str(output_file),
                "success": True
            }
        except Exception as e:
            return {
                "method": "PyPDF (raw)",
                "error": str(e),
                "success": False
            }

    def extract_with_langchain_pymupdf(self) -> Dict:
        start_time = time.time()
        try:
            loader = PyMuPDFLoader(str(self.pdf_path))
            documents = loader.load()

            text = ""
            for doc in documents:
                page_num = doc.metadata.get("page", "?")
                text += f"\n--- Page {page_num} ---\n"
                text += doc.page_content

            elapsed = time.time() - start_time

            output_file = self.output_dir / f"{self.pdf_name}_langchain_pymupdf.txt"
            output_file.write_text(text, encoding="utf-8")

            metadata_file = self.output_dir / f"{self.pdf_name}_langchain_pymupdf_metadata.txt"
            metadata_text = f"Total documents: {len(documents)}\n\n"
            for i, doc in enumerate(documents[:3]):
                metadata_text += f"Document {i+1} metadata:\n{doc.metadata}\n\n"
            metadata_file.write_text(metadata_text, encoding="utf-8")

            return {
                "method": "LangChain PyMuPDFLoader",
                "time": elapsed,
                "text_length": len(text),
                "documents": len(documents),
                "output_file": str(output_file),
                "metadata_file": str(metadata_file),
                "success": True
            }
        except Exception as e:
            return {
                "method": "LangChain PyMuPDFLoader",
                "error": str(e),
                "success": False
            }

    def extract_with_langchain_pdfplumber(self) -> Dict:
        start_time = time.time()
        try:
            loader = PDFPlumberLoader(str(self.pdf_path))
            documents = loader.load()

            text = ""
            for doc in documents:
                page_num = doc.metadata.get("page", "?")
                text += f"\n--- Page {page_num} ---\n"
                text += doc.page_content

            elapsed = time.time() - start_time

            output_file = self.output_dir / f"{self.pdf_name}_langchain_pdfplumber.txt"
            output_file.write_text(text, encoding="utf-8")

            metadata_file = self.output_dir / f"{self.pdf_name}_langchain_pdfplumber_metadata.txt"
            metadata_text = f"Total documents: {len(documents)}\n\n"
            for i, doc in enumerate(documents[:3]):
                metadata_text += f"Document {i+1} metadata:\n{doc.metadata}\n\n"
            metadata_file.write_text(metadata_text, encoding="utf-8")

            return {
                "method": "LangChain PDFPlumberLoader",
                "time": elapsed,
                "text_length": len(text),
                "documents": len(documents),
                "output_file": str(output_file),
                "metadata_file": str(metadata_file),
                "success": True
            }
        except Exception as e:
            return {
                "method": "LangChain PDFPlumberLoader",
                "error": str(e),
                "success": False
            }

    def extract_with_langchain_pypdf(self) -> Dict:
        start_time = time.time()
        try:
            loader = PyPDFLoader(str(self.pdf_path))
            documents = loader.load()

            text = ""
            for doc in documents:
                page_num = doc.metadata.get("page", "?")
                text += f"\n--- Page {page_num} ---\n"
                text += doc.page_content

            elapsed = time.time() - start_time

            output_file = self.output_dir / f"{self.pdf_name}_langchain_pypdf.txt"
            output_file.write_text(text, encoding="utf-8")

            metadata_file = self.output_dir / f"{self.pdf_name}_langchain_pypdf_metadata.txt"
            metadata_text = f"Total documents: {len(documents)}\n\n"
            for i, doc in enumerate(documents[:3]):
                metadata_text += f"Document {i+1} metadata:\n{doc.metadata}\n\n"
            metadata_file.write_text(metadata_text, encoding="utf-8")

            return {
                "method": "LangChain PyPDFLoader",
                "time": elapsed,
                "text_length": len(text),
                "documents": len(documents),
                "output_file": str(output_file),
                "metadata_file": str(metadata_file),
                "success": True
            }
        except Exception as e:
            return {
                "method": "LangChain PyPDFLoader",
                "error": str(e),
                "success": False
            }

    def run_all_extractions(self) -> List[Dict]:
        logger.info(f"Processing: {self.pdf_path.name}")

        results = []

        logger.info("  1/6 PyMuPDF (raw)...")
        results.append(self.extract_with_pymupdf_raw())

        logger.info("  2/6 PDFPlumber (raw)...")
        results.append(self.extract_with_pdfplumber_raw())

        logger.info("  3/6 PyPDF (raw)...")
        results.append(self.extract_with_pypdf_raw())

        logger.info("  4/6 LangChain PyMuPDFLoader...")
        results.append(self.extract_with_langchain_pymupdf())

        logger.info("  5/6 LangChain PDFPlumberLoader...")
        results.append(self.extract_with_langchain_pdfplumber())

        logger.info("  6/6 LangChain PyPDFLoader...")
        results.append(self.extract_with_langchain_pypdf())

        return results

def main():
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "data" / "reports"
    output_dir = project_root / "experiments" / "extraction_samples"

    pdf_files = list(reports_dir.rglob("*.pdf"))

    if not pdf_files:
        logger.error("No PDF files found")
        return

    logger.info(f"Found {len(pdf_files)} PDF files")

    selected_pdfs = pdf_files[:2]

    logger.info(f"Selected {len(selected_pdfs)} PDFs for comparison:")
    for pdf in selected_pdfs:
        logger.info(f"  - {pdf.name}")

    all_results = []

    for pdf_path in selected_pdfs:
        comparator = PDFExtractorComparison(pdf_path, output_dir)
        results = comparator.run_all_extractions()
        all_results.extend(results)
        logger.success(f"Completed: {pdf_path.name}\n")

    summary_file = output_dir / "comparison_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("PDF EXTRACTION COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            f.write(f"Method: {result.get('method', 'Unknown')}\n")
            if result.get('success'):
                f.write(f"  Time: {result.get('time', 0):.3f}s\n")
                f.write(f"  Text length: {result.get('text_length', 0):,} chars\n")
                if 'documents' in result:
                    f.write(f"  Documents: {result['documents']}\n")
                f.write(f"  Output: {result.get('output_file', 'N/A')}\n")
            else:
                f.write(f"  ERROR: {result.get('error', 'Unknown error')}\n")
            f.write("\n")

    logger.success(f"\nResults saved to: {output_dir}")
    logger.success(f"Summary: {summary_file}")

    logger.info("\nComparison by speed (fastest to slowest):")
    successful = [r for r in all_results if r.get('success')]
    sorted_by_time = sorted(successful, key=lambda x: x.get('time', float('inf')))
    for i, result in enumerate(sorted_by_time, 1):
        logger.info(f"  {i}. {result['method']}: {result['time']:.3f}s")

if __name__ == "__main__":
    main()
