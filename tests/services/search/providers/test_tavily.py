from unittest.mock import MagicMock, patch

import pytest

from src.services.search.providers.tavily import TavilyProvider


@pytest.fixture
def provider() -> TavilyProvider:
    return TavilyProvider(api_key="test-key")


def test_search_execution(provider: TavilyProvider) -> None:
    """Test basic Tavily search flow."""
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "answer": "Summary",
        "results": [
            {
                "title": "Result 1",
                "url": "http://1.com",
                "content": "Content 1",
            },
            {
                "title": "Result 2",
                "url": "http://2.com",
                "content": "Content 2",
            },
        ],
    }

    with patch(
        "src.services.search.providers.tavily.httpx.post",
        return_value=mock_http_response,
    ):
        response = provider.search("python testing", max_results=2)

    assert response.provider == "tavily"
    assert response.answer == "Summary"
    assert len(response.search_results) == 2
    assert response.search_results[0].title == "Result 1"
    assert response.search_results[0].url == "http://1.com"


def test_search_error_handling(provider: TavilyProvider) -> None:
    """Non-200 responses raise an exception."""
    mock_http_response = MagicMock()
    mock_http_response.status_code = 500
    mock_http_response.json.side_effect = ValueError("not json")
    mock_http_response.text = "Server error"

    with (
        patch(
            "src.services.search.providers.tavily.httpx.post",
            return_value=mock_http_response,
        ),
        pytest.raises(Exception, match="Tavily API error"),
    ):
        provider.search("fail test")
