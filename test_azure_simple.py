#!/usr/bin/env python3
"""
Simple test script for Azure OpenAI provider.
Run with: python test_azure_simple.py
"""

import os
import asyncio
from src.services.llm.factory import LLMFactory

async def test_azure():
    # Set environment variables (replace with your actual values)
    os.environ["AZURE_API_KEY"] = "your-azure-api-key"
    os.environ["AZURE_BASE_URL"] = "https://your-resource.openai.azure.com"
    os.environ["AZURE_DEPLOYMENT_NAME"] = "your-deployment-name"
    os.environ["AZURE_API_VERSION"] = "2024-02-01"

    try:
        # Create Azure provider
        provider = LLMFactory.create("azure", model_name="gpt-4")

        # Test completion
        prompt = "Hello, how are you?"
        response = await provider.complete(prompt)
        print(f"Response: {response}")

        # Test streaming
        print("Streaming response:")
        async for chunk in provider.stream(prompt):
            print(chunk, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_azure())
