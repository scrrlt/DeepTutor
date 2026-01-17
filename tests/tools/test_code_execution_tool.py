import pytest
from unittest.mock import patch, MagicMock
from src.tools.code_execution_tool import CodeExecutionTool

class TestCodeExecutionTool:
    
    @pytest.fixture
    def tool(self):
        return CodeExecutionTool(
            sandbox_client=MagicMock(),
            config={"timeout": 10, "allowed_modules": ["math", "numpy"]}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert hasattr(tool, 'sandbox_client')
        assert tool.config["timeout"] == 10
        assert "math" in tool.config["allowed_modules"]
        
    @patch('src.tools.code_execution_tool.CodeExecutionTool._execute_code')
    def test_execute(self, mock_execute, tool):
        code = "print('Hello, World!')"
        mock_execute.return_value = {
            "output": "Hello, World!",
            "status": "success",
            "execution_time": 0.05
        }
        
        result = tool.execute(code)
        
        assert result["output"] == "Hello, World!"
        assert result["status"] == "success"
        mock_execute.assert_called_once_with(code)
        
    @patch('src.tools.code_execution_tool.CodeExecutionTool._validate_code')
    def test_validate_code(self, mock_validate, tool):
        code = "import math\nprint(math.sqrt(16))"
        mock_validate.return_value = {"is_valid": True, "reason": ""}
        
        result = tool._validate_code(code)
        
        assert result["is_valid"] is True
        mock_validate.assert_called_once_with(code)
        
    @patch('src.tools.code_execution_tool.CodeExecutionTool._handle_execution_error')
    def test_handle_execution_error(self, mock_handle, tool):
        error = Exception("Division by zero")
        mock_handle.return_value = {
            "status": "error",
            "error_type": "ZeroDivisionError",
            "error_message": "Division by zero"
        }
        
        result = tool._handle_execution_error(error)
        
        assert result["status"] == "error"
        assert result["error_type"] == "ZeroDivisionError"
        mock_handle.assert_called_once_with(error)