"""
Tracing - Request ID and span management.
"""

from contextvars import ContextVar
import uuid

# Thread-local storage for request tracing
request_id = ContextVar("request_id", default=None)


def start_trace():
    """Start a new trace."""
    token = request_id.set(str(uuid.uuid4()))
    return token


def get_trace_id():
    """Get current trace ID."""
    return request_id.get()
