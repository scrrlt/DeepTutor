import pytest
from unittest.mock import patch, MagicMock
from src.tools.translation_tool import TranslationTool

class TestTranslationTool:
    
    @pytest.fixture
    def tool(self):
        return TranslationTool(
            translation_client=MagicMock(),
            config={"supported_languages": ["en", "es", "fr", "de", "zh"]}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert hasattr(tool, 'translation_client')
        assert "en" in tool.config["supported_languages"]
        assert "zh" in tool.config["supported_languages"]
        
    @patch('src.tools.translation_tool.TranslationTool._translate_text')
    def test_translate(self, mock_translate, tool):
        text = "Hello world"
        source_lang = "en"
        target_lang = "es"
        mock_translate.return_value = {
            "translated_text": "Hola mundo",
            "detected_language": "en",
            "confidence": 0.98
        }
        
        result = tool.translate(text, source_lang, target_lang)
        
        assert result["translated_text"] == "Hola mundo"
        assert result["confidence"] == 0.98
        mock_translate.assert_called_once_with(text, source_lang, target_lang)
        
    @patch('src.tools.translation_tool.TranslationTool._detect_language')
    def test_detect_language(self, mock_detect, tool):
        text = "Bonjour le monde"
        mock_detect.return_value = {"detected_language": "fr", "confidence": 0.95}
        
        result = tool.detect_language(text)
        
        assert result["detected_language"] == "fr"
        assert result["confidence"] == 0.95
        mock_detect.assert_called_once_with(text)
        
    @patch('src.tools.translation_tool.TranslationTool._is_language_supported')
    def test_is_language_supported(self, mock_supported, tool):
        mock_supported.side_effect = [True, False]
        
        result1 = tool._is_language_supported("en")
        result2 = tool._is_language_supported("sw")
        
        assert result1 is True
        assert result2 is False
        assert mock_supported.call_count == 2