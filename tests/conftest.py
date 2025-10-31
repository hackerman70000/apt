from pathlib import Path
from typing import List
import pytest
from langchain_core.documents import Document

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "reports").mkdir()
    (data_dir / "processed").mkdir()
    (data_dir / "chroma_db").mkdir()
    (data_dir / "logs").mkdir()
    return data_dir

@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_file = pdf_dir / "sample_report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nSample PDF content")
    return pdf_file

@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        Document(
            page_content="APT28 conducted spearphishing campaign using T1566.001 technique.",
            metadata={"filename": "apt28_report.pdf", "year": 2023, "page": 1}
        ),
        Document(
            page_content="Lazarus Group deployed custom malware in financial sector.",
            metadata={"filename": "lazarus_report.pdf", "year": 2024, "page": 1}
        ),
        Document(
            page_content="Analysis of HrServ webshell used by APT group.",
            metadata={"filename": "webshell_analysis.pdf", "year": 2023, "page": 5}
        ),
    ]

@pytest.fixture
def sample_chunks() -> List[Document]:
    return [
        Document(
            page_content="APT28 is a threat group attributed to Russian intelligence.",
            metadata={
                "filename": "apt28.pdf",
                "year": 2023,
                "apt_groups_mentioned": "APT28",
                "techniques_mentioned": "T1566.001"
            }
        ),
        Document(
            page_content="The group uses spearphishing as initial access vector.",
            metadata={
                "filename": "apt28.pdf",
                "year": 2023,
                "techniques_mentioned": "T1566"
            }
        ),
    ]

@pytest.fixture
def mock_config(tmp_data_dir: Path, monkeypatch):
    from apt import config

    monkeypatch.setattr(config.Config, "DATA_DIR", tmp_data_dir)
    monkeypatch.setattr(config.Config, "PROCESSED_DATA", tmp_data_dir / "processed")
    monkeypatch.setattr(config.Config, "CHROMA_DB", tmp_data_dir / "chroma_db")
    monkeypatch.setattr(config.Config, "REPORTS_DIR", tmp_data_dir / "reports")

    return config.Config
