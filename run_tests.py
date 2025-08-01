#!/usr/bin/env python3
"""
Test runner for the Simple Chatbot project.

Executes all tests and provides a comprehensive test report.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    from tests.test_simple import *
    
    print("Running SimpleChatbot Tests")
    print("===========================")

    # Test basic functionality
    test_basic_chatbot()

    if RAG_AVAILABLE:
        # Test RAG agent standalone
        test_rag_agent_standalone()
        
        # Test RAG with chatbot
        test_rag_chatbot()
        test_combined_chatbot()
    else:
        print("\nRAG dependencies not available. Install with:")
        print("pip install chromadb sentence-transformers")