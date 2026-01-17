import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.tools.document_loader_tool import DocumentLoaderTool

class TestDocumentLoaderTool:
    
    @pytest.fixture
    def tool(self):
        return DocumentLoaderTool(
            config={"supported_formats": ["pdf", "txt", "docx"]}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert "pdf" in tool.config["supported_formats"]
        assert "txt" in tool.config["supported_formats"]
        
    @patch('src.tools.document_loader_tool.DocumentLoaderTool._load_document')
    def test_load(self, mock_load, tool):
        file_path = "document.pdf"
        mock_load.return_value = {
            "content": "Document content",
            "metadata": {"pages": 5, "title": "Test Document"}
        }
        
        result = tool.load(file_path)
        
        assert result["content"] == "Document content"
        assert result["metadata"]["pages"] == 5
        mock_load.assert_called_once_with(file_path)
        
    @patch('src.tools.document_loader_tool.DocumentLoaderTool._detect_format')
    def test_detect_format(self, mock_detect, tool):
        file_path = "document.pdf"
        mock_detect.return_value = "pdf"
        
        result = tool._detect_format(file_path)
        
        assert result == "pdf"
        mock_detect.assert_called_once_with(file_path)
        
    @patch('builtins.open', new_callable=mock_open, read_data="Test content")
    def test_extract_text_from_txt(self, mock_file, tool):
        file_path = "document.txt"

        result = tool._extract_text_from_txt(file_path)
        
        assert result == "Test content"
        mock_file.assert_called_once_with(file_path, 'r', encoding='utf-8')