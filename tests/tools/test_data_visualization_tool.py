import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.tools.data_visualization_tool import DataVisualizationTool

class TestDataVisualizationTool:
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'x': np.random.rand(10),
            'y': np.random.rand(10),
            'category': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B']
        })
    
    @pytest.fixture
    def tool(self):
        return DataVisualizationTool(
            config={"default_style": "seaborn", "default_format": "png"}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert tool.config["default_style"] == "seaborn"
        assert tool.config["default_format"] == "png"
        
    @patch('src.tools.data_visualization_tool.DataVisualizationTool._create_plot')
    def test_create_plot(self, mock_plot, tool, sample_dataframe):
        plot_type = "scatter"
        x = "x"
        y = "y"
        mock_plot.return_value = {
            "plot_path": "/tmp/plot.png",
            "plot_type": "scatter",
            "dimensions": (800, 600)
        }
        
        result = tool.create_plot(sample_dataframe, plot_type, x=x, y=y)
        
        assert result["plot_path"] == "/tmp/plot.png"
        assert result["plot_type"] == "scatter"
        mock_plot.assert_called_once()
        
    @patch('src.tools.data_visualization_tool.DataVisualizationTool._create_histogram')
    def test_create_histogram(self, mock_histogram, tool, sample_dataframe):
        column = "x"
        bins = 10
        mock_histogram.return_value = {
            "plot_path": "/tmp/histogram.png",
            "bins": 10
        }
        
        result = tool.create_histogram(sample_dataframe, column, bins=bins)
        
        assert result["plot_path"] == "/tmp/histogram.png"
        assert result["bins"] == 10
        mock_histogram.assert_called_once()
        
    @patch('src.tools.data_visualization_tool.DataVisualizationTool._create_bar_chart')
    def test_create_bar_chart(self, mock_bar, tool, sample_dataframe):
        x = "category"
        y = "y"
        mock_bar.return_value = {
            "plot_path": "/tmp/bar.png",
            "categories": ["A", "B", "C"]
        }
        
        result = tool.create_bar_chart(sample_dataframe, x, y)
        
        assert result["plot_path"] == "/tmp/bar.png"
        assert "A" in result["categories"]
        mock_bar.assert_called_once()