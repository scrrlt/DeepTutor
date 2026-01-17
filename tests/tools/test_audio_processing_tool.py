import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.tools.audio_processing_tool import AudioProcessingTool

class TestAudioProcessingTool:
    
    @pytest.fixture
    def tool(self):
        return AudioProcessingTool(
            audio_engine=MagicMock(),
            config={"supported_formats": ["mp3", "wav", "ogg"], "max_duration": 300}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert hasattr(tool, 'audio_engine')
        assert "mp3" in tool.config["supported_formats"]
        assert tool.config["max_duration"] == 300
        
    @patch('src.tools.audio_processing_tool.AudioProcessingTool._transcribe_audio')
    def test_transcribe(self, mock_transcribe, tool):
        audio_path = "audio.mp3"
        mock_transcribe.return_value = {
            "text": "This is the transcribed text from the audio file.",
            "confidence": 0.92,
            "duration": 45.7
        }
        
        result = tool.transcribe(audio_path)
        
        assert "text" in result
        assert result["confidence"] == 0.92
        mock_transcribe.assert_called_once_with(audio_path)
        
    @patch('src.tools.audio_processing_tool.AudioProcessingTool._detect_language')
    def test_detect_language(self, mock_detect, tool):
        audio_path = "audio.mp3"
        mock_detect.return_value = {"language": "en", "confidence": 0.95}
        
        result = tool.detect_language(audio_path)
        
        assert result["language"] == "en"
        assert result["confidence"] == 0.95
        mock_detect.assert_called_once_with(audio_path)
        
    @patch('src.tools.audio_processing_tool.AudioProcessingTool._analyze_audio_quality')
    def test_analyze_audio_quality(self, mock_analyze, tool):
        audio_path = "audio.mp3"
        mock_analyze.return_value = {
            "noise_level": "low",
            "clarity": "high",
            "quality_score": 0.85
        }
        
        result = tool.analyze_audio_quality(audio_path)
        
        assert result["noise_level"] == "low"
        assert result["quality_score"] == 0.85
        mock_analyze.assert_called_once_with(audio_path)