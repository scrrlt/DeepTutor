"""
Fuzz tests for PDF parser (MinerU wrapper).
"""

from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import pytest

from src.tools.question import pdf_parser


def _fake_run(*args: Any, **kwargs: Any):
    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    return Result()


@settings(
    max_examples=30,
    deadline=1000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(text=st.text(min_size=0, max_size=200))
def test_parse_pdf_handles_random_paths(text: str, tmp_path: Path) -> None:
    """Ensure parser handles arbitrary paths without crashing."""
    fake_pdf = tmp_path / "file.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    with (
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setattr(pdf_parser.shutil, "which", lambda cmd: "/usr/bin/mineru")
        mp.setattr(pdf_parser.subprocess, "run", _fake_run)
        result = pdf_parser.parse_pdf_with_mineru(str(fake_pdf), text or None)
        assert isinstance(result, bool)


@settings(
    max_examples=30,
    deadline=1000,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(text=st.text(min_size=0, max_size=200))
def test_parse_pdf_rejects_non_pdf_paths(text: str, tmp_path: Path) -> None:
    """Non-PDF input should return False without crashing."""
    fake_file = tmp_path / "file.txt"
    fake_file.write_text("not a pdf")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pdf_parser.shutil, "which", lambda cmd: "/usr/bin/mineru")
        mp.setattr(pdf_parser.subprocess, "run", _fake_run)
        result = pdf_parser.parse_pdf_with_mineru(str(fake_file), text or None)
        assert result is False
