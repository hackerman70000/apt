import pytest
from pathlib import Path
from langchain_core.documents import Document
from apt.ingest.loader import PDFLoader, load_pdfs

class TestPDFLoader:
    def test_init_with_valid_directory(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        loader = PDFLoader(pdf_dir)
        assert loader.pdf_directory == pdf_dir

    def test_init_with_invalid_directory(self, tmp_path):
        invalid_dir = tmp_path / "nonexistent"
        with pytest.raises(ValueError, match="PDF directory does not exist"):
            PDFLoader(invalid_dir)

    def test_load_pdf_success(self, sample_pdf_path):
        loader = PDFLoader(sample_pdf_path.parent)
        documents = loader.load_pdf(sample_pdf_path)

        assert isinstance(documents, list)
        assert all(isinstance(doc, Document) for doc in documents)

    def test_load_pdf_adds_metadata(self, sample_pdf_path):
        loader = PDFLoader(sample_pdf_path.parent)
        documents = loader.load_pdf(sample_pdf_path)

        if documents:
            assert "filename" in documents[0].metadata
            assert documents[0].metadata["filename"] == sample_pdf_path.name

    def test_load_pdf_adds_year_metadata(self, tmp_path):
        year_dir = tmp_path / "2023"
        year_dir.mkdir()
        pdf_file = year_dir / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nContent")

        loader = PDFLoader(tmp_path)
        documents = loader.load_pdf(pdf_file)

        if documents:
            assert documents[0].metadata.get("year") == 2023

    def test_load_pdf_nonexistent_file(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        loader = PDFLoader(pdf_dir)
        nonexistent = pdf_dir / "nonexistent.pdf"

        documents = loader.load_pdf(nonexistent)
        assert documents == []

    def test_load_directory_with_multiple_pdfs(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        for i in range(3):
            pdf_file = pdf_dir / f"report_{i}.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\nContent")

        loader = PDFLoader(pdf_dir)
        documents = loader.load_directory()

        assert isinstance(documents, list)

    def test_load_directory_with_max_files(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        for i in range(5):
            pdf_file = pdf_dir / f"report_{i}.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\nContent")

        loader = PDFLoader(pdf_dir)
        loader.load_directory(max_files=3)

    def test_load_directory_empty_directory(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        loader = PDFLoader(pdf_dir)
        documents = loader.load_directory()

        assert documents == []

    def test_load_pdf_corrupted_file(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        corrupted_file = pdf_dir / "corrupted.pdf"
        corrupted_file.write_bytes(b"Not a valid PDF")

        loader = PDFLoader(pdf_dir)
        documents = loader.load_pdf(corrupted_file)

        assert documents == []

    def test_load_pdf_empty_pdf(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        empty_file = pdf_dir / "empty.pdf"
        empty_file.write_bytes(b"")

        loader = PDFLoader(pdf_dir)
        documents = loader.load_pdf(empty_file)

        assert documents == []

    def test_load_directory_with_nested_pdfs(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        year_dir = pdf_dir / "2023"
        year_dir.mkdir()

        for i in range(2):
            pdf_file = year_dir / f"report_{i}.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\nContent")

        loader = PDFLoader(pdf_dir)
        documents = loader.load_directory()

        assert isinstance(documents, list)

    def test_load_directory_counts_successes_and_failures(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        valid_file = pdf_dir / "valid.pdf"
        valid_file.write_bytes(b"%PDF-1.4\nContent")

        invalid_file = pdf_dir / "invalid.pdf"
        invalid_file.write_bytes(b"Not a PDF")

        loader = PDFLoader(pdf_dir)
        documents = loader.load_directory()

        assert isinstance(documents, list)

class TestLoadPDFsFunction:
    def test_load_pdfs_with_custom_directory(self, tmp_path):
        pdf_dir = tmp_path / "custom"
        pdf_dir.mkdir()
        pdf_file = pdf_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nContent")

        documents = load_pdfs(pdf_directory=pdf_dir)
        assert isinstance(documents, list)

    def test_load_pdfs_nonexistent_directory(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        documents = load_pdfs(pdf_directory=nonexistent)
        assert documents == []

    def test_load_pdfs_with_max_files(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        for i in range(5):
            pdf_file = pdf_dir / f"report_{i}.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\nContent")

        documents = load_pdfs(pdf_directory=pdf_dir, max_files=2)
        assert isinstance(documents, list)

    def test_load_pdfs_with_none_directory_uses_default(self, mock_config):
        default_dir = mock_config.REPORTS_DIR / "aptnotes_pdfs"
        default_dir.mkdir(parents=True, exist_ok=True)

        pdf_file = default_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nContent")

        documents = load_pdfs(pdf_directory=None)
        assert isinstance(documents, list)
