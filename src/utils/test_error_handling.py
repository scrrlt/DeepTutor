#!/usr/bin/env python
"""Manual test script for error handling features"""

import asyncio


# Test 1: Error message formatting
def test_error_formatting():
    from src.utils.error_utils import format_exception_message

    # Test basic exception
    try:
        raise ValueError("Test error message")
    except Exception as e:
        msg = format_exception_message(e)
        assert "Test error message" in msg
        print("✓ Basic error formatting works")

    # Test with OpenAI-like error
    try:
        from openai import RateLimitError

        raise RateLimitError("Rate limit exceeded", response=None, body=None)
    except Exception as e:
        msg = format_exception_message(e)
        assert "rate limit" in msg.lower()
        print("✓ OpenAI error formatting works")


# Test 2: Document validation
def test_document_validation():
    from src.utils.document_validator import DocumentValidator

    # Test file size validation
    try:
        DocumentValidator.validate_upload_safety("test.pdf", 0)
        assert False, "Should raise ValueError for empty file"
    except ValueError as e:
        assert "empty" in str(e).lower()
        print("✓ Empty file validation works")

    # Test filename sanitization
    try:
        DocumentValidator.validate_upload_safety("../../etc/passwd", 1024)
        assert False, "Should raise ValueError for path traversal"
    except ValueError:
        print("✓ Path traversal protection works")

    # Test valid filename
    safe_name = DocumentValidator.validate_upload_safety("test.pdf", 1024)
    assert safe_name == "test.pdf"
    print("✓ Valid filename passes")


# Test 3: Retry decorator
async def test_retry_decorator():
    from src.agents.solve.utils.error_handler import ParseError, retry_on_parse_error

    attempt_count = 0

    @retry_on_parse_error(max_retries=2, delay=0.1)
    async def failing_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ParseError("Temporary failure")
        return "success"

    result = await failing_function()
    assert result == "success"
    assert attempt_count == 3
    print(f"✓ Retry decorator works (retried {attempt_count} times)")


# Test 4: Validation functions
async def test_validation_functions():
    from src.agents.solve.utils.error_handler import (
        ParseError,
        validate_investigate_output,
    )

    # Valid investigate output
    valid_output = {
        "reasoning": "Need to search for information",
        "tools": [{"tool_type": "rag_naive", "query": "What is AI?", "identifier": ""}],
    }
    assert await validate_investigate_output(valid_output)
    print("✓ Valid investigate output passes")

    # Invalid - missing required field
    try:
        await validate_investigate_output({"tools": []})
        assert False, "Should raise ParseError"
    except ParseError as e:
        assert "reasoning" in str(e)
        print("✓ Missing field detection works")

    # Invalid - none tool with others
    try:
        invalid_output = {
            "reasoning": "test",
            "tools": [
                {"tool_type": "none", "query": "", "identifier": ""},
                {"tool_type": "rag_naive", "query": "test", "identifier": ""},
            ],
        }
        await validate_investigate_output(invalid_output)
        assert False, "Should raise ParseError for none tool constraint"
    except ParseError as e:
        assert "none" in str(e).lower()
        print("✓ None tool constraint validation works")


if __name__ == "__main__":
    print("Running error handling tests...\n")

    test_error_formatting()
    test_document_validation()
    asyncio.run(test_retry_decorator())
    asyncio.run(test_validation_functions())

    print("\n✅ All manual tests passed!")
