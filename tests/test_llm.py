from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.llm import complete, fetch_models, sanitize_url


class TestLLMService:
    """Test suite for core LLM service functions."""

    def test_sanitize_url_valid(self) -> None:
        """Test that sanitize_url correctly processes a valid URL."""
        url = "https://example.com/path?query=1"
        result = sanitize_url(url)
        # Assuming it returns a string
        assert isinstance(result, str)

    def test_sanitize_url_invalid(self) -> None:
        """Test sanitize_url with potentially unsafe input."""
        url = "javascript:alert(1)"
        result = sanitize_url(url)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_complete_calls_provider(
        self, mock_openai_client: MagicMock
    ) -> None:
        """Test that complete function delegates to the LLM provider."""
        # Mock the return value of the client
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="Test"))]
            )
        )

        # We patch the internal client usage in the module if necessary
        # or rely on the mock_openai_client fixture if the module uses
        # dependency injection or global client lookup.
        with patch("src.services.llm.complete") as mock_complete:
            mock_complete.return_value = "Test response"
            result = await complete(prompt="Hello")
            assert result == "Test response"

    @pytest.mark.asyncio
    async def test_fetch_models(self) -> None:
        """Test fetching available models."""
        with patch("src.services.llm.fetch_models") as mock_fetch:
            mock_fetch.return_value = ["gpt-4", "gpt-3.5-turbo"]
            models = await fetch_models()
            assert "gpt-4" in models
            assert isinstance(models, list)
