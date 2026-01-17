import pytest
from unittest.mock import patch, MagicMock
from src.agents.question.example import ExampleQuestionProcessor

class TestExampleQuestionProcessor:
    
    @pytest.fixture
    def processor(self):
        return ExampleQuestionProcessor()
    
    def test_initialization(self, processor):
        assert processor is not None
        
    @patch('src.agents.question.example.ExampleQuestionProcessor._process_query')
    def test_process_question(self, mock_process, processor):
        mock_process.return_value = {"answer": "Test answer"}
        result = processor.process_question("What is testing?")
        assert result == {"answer": "Test answer"}
        mock_process.assert_called_once_with("What is testing?")
    
    @patch('src.agents.question.example.ExampleQuestionProcessor._get_context')
    def test_get_context(self, mock_get_context, processor):
        mock_get_context.return_value = ["Context 1", "Context 2"]
        context = processor._get_context("query")
        assert context == ["Context 1", "Context 2"]
        
    @patch('src.agents.question.example.ExampleQuestionProcessor._format_output')
    def test_format_output(self, mock_format, processor):
        mock_format.return_value = "Formatted answer"
        output = processor.format_answer("Raw answer")
        assert output == "Formatted answer"
        mock_format.assert_called_once_with("Raw answer")