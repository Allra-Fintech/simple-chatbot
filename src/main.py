#!/usr/bin/env python3
"""
Simple demo showing RAG (Retrieval Augmented Generation) and Function Calling
"""

import ollama
from .function_agent import simple_function_call
from .rag_agent import simple_rag_query


def demo_function_calling():
    """Demo: Function calling for time and math"""
    print("\n=== FUNCTION CALLING DEMO ===")
    print("Shows how LLM can call external functions")

    queries = [
        "What time is it?",
        "Calculate 15 * 7 + 23",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = simple_function_call(query)
        print(f"Bot: {response}")


def demo_rag():
    """Demo: RAG with document retrieval"""
    print("\n=== RAG DEMO ===")
    print("Shows how LLM can answer based on your documents")

    # Add some sample documents
    print("\nAdding sample documents...")
    docs = [
        "Python is a programming language created by Guido van Rossum in 1991.",
        "Machine learning is a subset of AI that learns from data.",
        "ChromaDB is a vector database for AI applications.",
    ]

    for doc in docs:
        print(f"Added: {doc[:50]}...")

    # Query the documents
    queries = [
        "Who created Python?",
        "What is machine learning?",
        "Tell me about ChromaDB",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = simple_rag_query(query, docs)
        print(f"Bot: {response}")


def main():
    """Main demo runner"""
    print("RAG and Function Calling Demo")
    print("=" * 40)

    # Check Ollama connection
    try:
        ollama.list()
        print("✓ Ollama connected")
    except Exception:
        print("✗ Ollama not running. Start with: ollama serve")
        return

    # Run demos
    demo_function_calling()
    demo_rag()

    print("\n=== DEMO COMPLETE ===")
    print("This shows the basic concepts of RAG and function calling.")
    print("RAG: Retrieves relevant documents to answer questions")
    print("Functions: Allows LLM to call external tools (time, math, etc.)")


if __name__ == "__main__":
    main()
