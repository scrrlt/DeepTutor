import pytest
from unittest.mock import patch, MagicMock
from src.tools.database_tool import DatabaseTool

class TestDatabaseTool:
    
    @pytest.fixture
    def tool(self):
        db_connection = MagicMock()
        return DatabaseTool(
            db_connection=db_connection,
            config={"timeout": 30, "max_rows": 1000}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert hasattr(tool, 'db_connection')
        assert tool.config["timeout"] == 30
        assert tool.config["max_rows"] == 1000
        
    @patch('src.tools.database_tool.DatabaseTool._execute_query')
    def test_execute_query(self, mock_execute, tool):
        query = "SELECT * FROM users"
        mock_execute.return_value = {
            "rows": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}],
            "row_count": 2,
            "columns": ["id", "name"]
        }
        
        result = tool.execute_query(query)
        
        assert len(result["rows"]) == 2
        assert result["rows"][0]["name"] == "John"
        assert result["columns"] == ["id", "name"]
        mock_execute.assert_called_once_with(query)
        
    @patch('src.tools.database_tool.DatabaseTool._get_table_schema')
    def test_get_table_schema(self, mock_schema, tool):
        table_name = "users"
        mock_schema.return_value = {
            "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "name", "type": "VARCHAR", "nullable": True}
            ],
            "primary_key": ["id"]
        }
        
        result = tool.get_table_schema(table_name)
        
        assert len(result["columns"]) == 2
        assert result["columns"][0]["name"] == "id"
