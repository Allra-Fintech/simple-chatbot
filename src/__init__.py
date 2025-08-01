"""
Simple RAG and Function Calling Demo

Educational project showing how RAG and function calling work with LLMs.
"""

__version__ = "1.0.0"

from .function_agent import simple_function_call
from .rag_agent import simple_rag_query

__all__ = ["simple_function_call", "simple_rag_query"]
