"""
Simple Chatbot with Function Calling and RAG

A Python-based chatbot that uses Ollama for local LLM inference with optional 
function calling and RAG (Retrieval Augmented Generation) capabilities.
"""

__version__ = "1.0.0"
__author__ = "Simple Chatbot Team"

from .main import SimpleChatbot
from .function_agent import FunctionCallerAgent
from .rag_agent import RAGAgent

__all__ = ["SimpleChatbot", "FunctionCallerAgent", "RAGAgent"]