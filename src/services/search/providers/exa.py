# -*- coding: utf-8 -*-
"""
Exa Neural Search Provider

API Docs: https://exa.ai/docs/reference/search
Endpoint: https://api.exa.ai/search

Features:
- Embeddings-based neural search (finds semantically similar content)
- Multiple search types: auto, neural, keyword
- Category filtering: research paper, news, company, people, github, tweet, pdf
- Date filtering (published date and crawl date)
- Domain include/exclude lists
- Full text extraction with highlights and summaries
- Cost tracking in response

Pricing:
- Neural search (1-25 results): $0.005/request
- Neural search (26-100 results): $0.025/request
- Content text/highlight/summary: $0.001/page
"""

from datetime import datetime
from typing import Any

import httpx

from ..base import BaseSearchProvider
from ..types import Citation, SearchResult, WebSearchResponse
from . import register_provider


@register_provider("exa")
class ExaProvider(BaseSearchProvider):
    """Exa neural/embeddings-based search provider"""

    display_name = "Exa"
    description = "Neural/embeddings search"
    supports_answer = True  # Provides summaries and context
    BASE_URL = "https://api.exa.ai/search"

    async def search(
        self,
        query: str,
        search_type: str = "auto",  # auto, neural, keyword
        num_results: int = 10,
        include_text: bool = True,
        include_highlights: bool = True,
        include_summary: bool = True,
        max_characters: int | None = None,
        category: str | None = None,  # research paper, news, company, etc.
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_published_date: str | None = None,  # ISO format
        end_published_date: str | None = None,
        timeout: int = 60,
        **kwargs: Any,
    ) -> WebSearchResponse:
        """
        Perform a search via the Exa API and return a standardized WebSearchResponse containing results, citations, and optional summaries.
        
        Parameters:
            query (str): Search query string.
            search_type (str): One of "auto", "neural", or "keyword" to select the search mode.
            max_characters (int | None): Maximum number of characters to include for each result's text when `include_text` is true; if None, the full text may be returned.
            category (str | None): Optional category filter (e.g., "research paper", "news", "company").
            include_domains (list[str] | None): Optional list of domains to restrict results to.
            exclude_domains (list[str] | None): Optional list of domains to exclude from results.
            start_published_date (str | None): Optional ISO-format start date to filter published results (inclusive).
            end_published_date (str | None): Optional ISO-format end date to filter published results (inclusive).
            timeout (int): Request timeout in seconds.
        
        Returns:
            WebSearchResponse: Standardized response containing the original query, assembled answer (combined summaries if present), provider metadata, list of citations, list of SearchResult items, and request metadata.
        """
        self.logger.debug(f"Calling Exa API type={search_type}, num_results={num_results}")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

        # Build contents configuration
        contents: dict[str, Any] = {}
        if include_text:
            contents["text"] = {"maxCharacters": max_characters} if max_characters else True
        if include_highlights:
            contents["highlights"] = True
        if include_summary:
            contents["summary"] = True

        payload: dict[str, Any] = {
            "query": query,
            "type": search_type,
            "numResults": num_results,
            "contents": contents,
        }

        if category:
            payload["category"] = category
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains
        if start_published_date:
            payload["startPublishedDate"] = start_published_date
        if end_published_date:
            payload["endPublishedDate"] = end_published_date

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.BASE_URL, headers=headers, json=payload, timeout=timeout)
            except httpx.RequestError as e:
                self.logger.error(f"Exa request failed: {e}")
                raise Exception(f"Exa request failed: {e}")

        if response.status_code != 200:
            try:
                error_data = response.json() if response.text else {}
            except Exception:
                error_data = {}
            self.logger.error(f"Exa API error: {response.status_code}")
            raise Exception(
                f"Exa API error: {response.status_code} - {error_data.get('error', response.text)}"
            )

        try:
            data = response.json()
        except Exception as e:
            raise Exception(f"Failed to parse Exa API response: {e}")
        self.logger.debug(f"Exa returned {len(data.get('results', []))} results")

        # Build answer from summaries
        summaries = []
        citations: list[Citation] = []
        search_results: list[SearchResult] = []

        for i, result in enumerate(data.get("results", []), 1):
            # Extract summary for answer
            summary = result.get("summary", "")
            if summary:
                summaries.append(f"[{i}] {summary}")

            # Build search result
            sr = SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=summary or result.get("text", "")[:500],
                date=result.get("publishedDate", ""),
                source=result.get("author", ""),
                content=result.get("text", ""),
                score=result.get("score", 0.0),
            )
            search_results.append(sr)

            # Build citation
            citations.append(
                Citation(
                    id=i,
                    reference=f"[{i}]",
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    snippet=summary or result.get("text", "")[:500],
                    date=result.get("publishedDate", ""),
                    source=result.get("author", ""),
                    content=result.get("text", ""),
                )
            )

        # Combine summaries as answer
        answer = "\n\n".join(summaries) if summaries else ""

        # Build metadata
        metadata: dict[str, Any] = {
            "finish_reason": "stop",
            "search_type": search_type,
            "autoprompt_string": data.get("autopromptString", ""),
        }

        # Add cost info if available
        if data.get("costDollars"):
            metadata["cost_dollars"] = data["costDollars"]

        response_obj = WebSearchResponse(
            query=query,
            answer=answer,
            provider="exa",
            timestamp=datetime.now().isoformat(),
            model=f"exa-{search_type}",
            citations=citations,
            search_results=search_results,
            usage={},
            metadata=metadata,
        )

        return response_obj