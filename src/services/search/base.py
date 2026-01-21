# -*- coding: utf-8 -*-
"""
Web Search Base Provider - Abstract base class for all search providers

This module defines the BaseSearchProvider class that all search providers must inherit from.
All providers use a unified SEARCH_API_KEY environment variable.
"""

from abc import ABC, abstractmethod
import os
from typing import Any

from src.logging import get_logger

from .types import WebSearchResponse

# Unified API key environment variable
SEARCH_API_KEY_ENV = "SEARCH_API_KEY"


class BaseSearchProvider(ABC):
    """Abstract base class for search providers.

    All providers use a unified SEARCH_API_KEY environment variable.
    Each provider has its own BASE_URL defined as a class constant.
    """

    name: str = "base"
    display_name: str = "Base Provider"
    description: str = ""
    requires_api_key: bool = True
    supports_answer: bool = False  # Whether provider generates LLM answers
    BASE_URL: str = ""  # Each provider defines its own endpoint

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        """
        Initialize the provider.

        Args:
            api_key: API key for the provider. If not provided, will be read from SEARCH_API_KEY.
            **kwargs: Additional configuration options.
        """
        self.logger = get_logger(f"Search.{self.__class__.__name__}", level="INFO")
        self.api_key = api_key or self._get_api_key()
        self.config = kwargs

    def _get_api_key(self) -> str:
        """
        Retrieve the provider's API key from the SEARCH_API_KEY environment variable.
        
        Reads the environment variable named by SEARCH_API_KEY_ENV and returns its value.
        If the provider requires an API key and the environment variable is missing or empty,
        a ValueError is raised.
        
        Returns:
            str: The API key string from the environment (may be empty if not required).
        
        Raises:
            ValueError: If `requires_api_key` is True and the environment variable is not set or empty.
        """
        key = os.environ.get(SEARCH_API_KEY_ENV, "")
        if self.requires_api_key and not key:
            raise ValueError(
                f"{self.name} requires {SEARCH_API_KEY_ENV} environment variable. "
                f"Please set it before using this provider."
            )
        return key

    @abstractmethod
    async def search(self, query: str, **kwargs: Any) -> WebSearchResponse:
        """
        Perform a web search for the given query using this provider and return a standardized response.
        
        Parameters:
            query (str): The search query string.
            **kwargs: Provider-specific options that modify the search behavior (e.g., pagination, filters, locale).
        
        Returns:
            WebSearchResponse: A standardized response containing search results and associated metadata.
        """
        pass

    def is_available(self) -> bool:
        """
        Check if provider is available (dependencies installed, API key set).

        Returns:
            bool: True if provider is available, False otherwise.
        """
        try:
            if self.requires_api_key:
                key = self.api_key or os.environ.get(SEARCH_API_KEY_ENV, "")
                if not key:
                    return False
            return True
        except (ValueError, ImportError):
            return False


__all__ = ["BaseSearchProvider", "SEARCH_API_KEY_ENV"]