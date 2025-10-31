import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
import sys
import importlib.util

runner = CliRunner()

class TestExtractCLI:
    def test_extract_tool_exists(self):
        extract_path = Path(__file__).parent.parent.parent / "tools" / "extract"
        assert extract_path.exists()
        assert extract_path.is_file()

    def test_extract_has_uv_shebang(self):
        extract_path = Path(__file__).parent.parent.parent / "tools" / "extract"
        with open(extract_path) as f:
            first_line = f.readline()
            assert "#!/usr/bin/env -S uv run" in first_line

class TestEmbedCLI:
    def test_embed_tool_exists(self):
        embed_path = Path(__file__).parent.parent.parent / "tools" / "embed"
        assert embed_path.exists()
        assert embed_path.is_file()

    def test_embed_has_uv_shebang(self):
        embed_path = Path(__file__).parent.parent.parent / "tools" / "embed"
        with open(embed_path) as f:
            first_line = f.readline()
            assert "#!/usr/bin/env -S uv run" in first_line

class TestQueryCLI:
    def test_query_tool_exists(self):
        query_path = Path(__file__).parent.parent.parent / "tools" / "query"
        assert query_path.exists()
        assert query_path.is_file()

    def test_query_has_uv_shebang(self):
        query_path = Path(__file__).parent.parent.parent / "tools" / "query"
        with open(query_path) as f:
            first_line = f.readline()
            assert "#!/usr/bin/env -S uv run" in first_line

class TestFetchCLI:
    def test_fetch_tool_exists(self):
        fetch_path = Path(__file__).parent.parent.parent / "tools" / "fetch"
        assert fetch_path.exists()
        assert fetch_path.is_file()

    def test_fetch_has_uv_shebang(self):
        fetch_path = Path(__file__).parent.parent.parent / "tools" / "fetch"
        with open(fetch_path) as f:
            first_line = f.readline()
            assert "#!/usr/bin/env -S uv run" in first_line

class TestToolsExecutable:
    def test_all_tools_executable(self):
        tools_dir = Path(__file__).parent.parent.parent / "tools"
        tool_files = ["extract", "embed", "query", "fetch"]

        for tool in tool_files:
            tool_path = tools_dir / tool
            assert tool_path.exists(), f"{tool} does not exist"
            assert tool_path.stat().st_mode & 0o111, f"{tool} is not executable"

class TestExtractExecution:
    def test_extract_help_command(self):
        extract_path = Path(__file__).parent.parent.parent / "tools" / "extract"

        spec = importlib.util.spec_from_file_location("extract", extract_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["extract"] = module

            with patch('sys.argv', ['extract', '--help']):
                try:
                    spec.loader.exec_module(module)
                except SystemExit as e:
                    assert e.code in [0, None]

    def test_extract_imports_correctly(self):
        extract_path = Path(__file__).parent.parent.parent / "tools" / "extract"

        with open(extract_path) as f:
            content = f.read()

        assert "from apt.ingest import load_pdfs, chunk_documents" in content
        assert "from apt.config import Config" in content
        assert "import typer" in content

class TestEmbedExecution:
    def test_embed_imports_correctly(self):
        embed_path = Path(__file__).parent.parent.parent / "tools" / "embed"

        with open(embed_path) as f:
            content = f.read()

        assert "from apt.store import ChromaManager" in content
        assert "from apt.config import Config" in content
        assert "import typer" in content

class TestQueryExecution:
    def test_query_imports_correctly(self):
        query_path = Path(__file__).parent.parent.parent / "tools" / "query"

        with open(query_path) as f:
            content = f.read()

        assert "from apt.store import ChromaManager" in content
        assert "from apt.retrieval import create_rag_chain" in content
        assert "import typer" in content

    def test_query_has_main_command(self):
        query_path = Path(__file__).parent.parent.parent / "tools" / "query"

        with open(query_path) as f:
            content = f.read()

        assert "@app.command()" in content
        assert "def main(" in content
        assert "question:" in content

class TestFetchExecution:
    def test_fetch_imports_correctly(self):
        fetch_path = Path(__file__).parent.parent.parent / "tools" / "fetch"

        with open(fetch_path) as f:
            content = f.read()

        assert "from apt.config import Config" in content
        assert "import typer" in content
        assert "import requests" in content
