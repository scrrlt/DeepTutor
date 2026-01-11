#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constants for DeepTutor
"""

# Valid tools for investigate agent
VALID_INVESTIGATE_TOOLS = ["rag_naive", "rag_hybrid", "web_search", "query_item", "none"]

# Valid tools for solve agent
VALID_SOLVE_TOOLS = [
    "web_search",
    "code_execution",
    "rag_naive",
    "rag_hybrid",
    "query_item",
    "none",
]
