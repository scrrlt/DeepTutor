import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.tools.question import pdf_parser

@pytest.fixture
def mock_mineru_check():
    with patch("src.tools.question.pdf_parser.shutil.which") as mock_which:
        mock_which.return_value = "/usr/bin/mineru"
        yield mock_which

@pytest.fixture
def mock_subprocess():
    with patch("src.tools.question.pdf_parser.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        yield mock_run

def test_get_mineru_command_finds_magic_pdf():
    with patch("src.tools.question.pdf_parser.shutil.which") as mock_which:
        mock_which.side_effect = lambda cmd: "/bin/magic-pdf" if cmd == "magic-pdf" else None
        assert pdf_parser._get_mineru_command() == "magic-pdf"

def test_get_mineru_command_finds_mineru():
    with patch("src.tools.question.pdf_parser.shutil.which") as mock_which:
        mock_which.side_effect = lambda cmd: "/bin/mineru" if cmd == "mineru" else None
        assert pdf_parser._get_mineru_command() == "mineru"

def test_get_mineru_command_fails():
    with patch("src.tools.question.pdf_parser.shutil.which", return_value=None):
        assert pdf_parser._get_mineru_command() is None

def test_parse_pdf_mineru_missing(tmp_path):
    with patch("src.tools.question.pdf_parser._get_mineru_command", return_value=None):
        assert pdf_parser.parse_pdf_with_mineru("test.pdf") is False

def test_parse_pdf_invalid_pdf_path(mock_mineru_check):
    assert pdf_parser.parse_pdf_with_mineru("non_existent.pdf") is False
    assert pdf_parser.parse_pdf_with_mineru("exist.txt") is False

def test_parse_pdf_output_creation_failure(tmp_path, mock_mineru_check):
    pdf = tmp_path / "test.pdf"
    pdf.touch()
    
    # Mock mkdir to raise OSError
    with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
         assert pdf_parser.parse_pdf_with_mineru(str(pdf), output_base_dir="/root/restricted") is False

def test_parse_pdf_success(tmp_path, mock_mineru_check, mock_subprocess):
    pdf = tmp_path / "test.pdf"
    pdf.touch()
    output_dir = tmp_path / "output"
    
    def side_effect(cmd, **kwargs):
        # The output dir is the last argument or -o flag
        # cmd = [mineru, -p, pdf, -o, output_path]
        output_path = Path(cmd[4])
        # Create a dummy subdir as if mineru did it
        (output_path / "test_subdir").mkdir(parents=True, exist_ok=True)
        return MagicMock(returncode=0)

    mock_subprocess.side_effect = side_effect
    
    assert pdf_parser.parse_pdf_with_mineru(str(pdf), output_base_dir=str(output_dir)) is True
    
    # Verify subprocess call
    mock_subprocess.assert_called_once()
    cmd = mock_subprocess.call_args[0][0]
    # It might resolve to magic-pdf or mineru depending on what shutil.which returns first in loop
    assert any(x in cmd[0] for x in ["mineru", "magic-pdf"])
    assert str(pdf) in cmd

def test_parse_pdf_backup_rotation(tmp_path, mock_mineru_check, mock_subprocess):
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    output_base = tmp_path / "output"
    output_base.mkdir()
    
    # Create existing output
    existing_output = output_base / "doc"
    existing_output.mkdir()
    (existing_output / "old_file.txt").touch()

    def side_effect(cmd, **kwargs):
        output_path = Path(cmd[4])
        # Create a dummy subdir as if mineru did it. Use matching name 'doc' if possible or random
        (output_path / "doc").mkdir(parents=True, exist_ok=True)
        return MagicMock(returncode=0)

    mock_subprocess.side_effect = side_effect
    
    with patch("src.tools.question.pdf_parser.shutil.move") as mock_move:
         assert pdf_parser.parse_pdf_with_mineru(str(pdf), output_base_dir=str(output_base)) is True
         # Called twice: once for backup, once for moving result
         assert mock_move.call_count == 2
