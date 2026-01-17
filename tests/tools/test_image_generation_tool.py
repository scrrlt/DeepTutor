import pytest
from unittest.mock import patch, MagicMock
from src.tools.image_generation_tool import ImageGenerationTool

class TestImageGenerationTool:
    
    @pytest.fixture
    def tool(self):
        return ImageGenerationTool(
            image_api_client=MagicMock(),
            config={"resolution": "512x512", "style": "realistic"}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert hasattr(tool, 'image_api_client')
        assert tool.config["resolution"] == "512x512"
        assert tool.config["style"] == "realistic"
        
    @patch('src.tools.image_generation_tool.ImageGenerationTool._generate_image')
    def test_generate(self, mock_generate, tool):
        prompt = "A sunset over mountains"
        mock_generate.return_value = {
            "image_url": "https://example.com/image.png",
            "generation_id": "gen_123456"
        }
        
        result = tool.generate(prompt)
        
        assert "image_url" in result
        assert result["image_url"] == "https://example.com/image.png"
        mock_generate.assert_called_once_with(prompt)
        
    @patch('src.tools.image_generation_tool.ImageGenerationTool._enhance_prompt')
    def test_enhance_prompt(self, mock_enhance, tool):
        prompt = "sunset mountains"
        mock_enhance.return_value = "A beautiful sunset over majestic mountains with golden light"
        
        result = tool.enhance_prompt(prompt)
        
        assert result == "A beautiful sunset over majestic mountains with golden light"
        mock_enhance.assert_called_once_with(prompt)
        
    @patch('src.tools.image_generation_tool.ImageGenerationTool._validate_image')
    def test_validate_image(self, mock_validate, tool):
        image_data = {"url": "https://example.com/image.png"}
        mock_validate.return_value = {"is_valid": True, "reason": ""}
        
        result = tool._validate_image(image_data)
        
        assert result["is_valid"] is True
        mock_validate.assert_called_once_with(image_data)