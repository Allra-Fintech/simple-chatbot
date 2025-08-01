#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import SimpleChatbot

try:
    from src.rag_agent import RAGAgent
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


def test_basic_chatbot():
    """Test the basic chatbot functionality"""
    print("Testing SimpleChatbot (Basic Mode)...")

    chatbot = SimpleChatbot(use_function_agent=True, use_rag_agent=False)

    test_queries = [
        "What time is it?",
        "Calculate 25 + 17",
        "What's the weather like?",
        "Hello, how are you?",
    ]

    for query in test_queries:
        print("=======================================================")
        print(f"\nQuery: {query}")
        try:
            response = chatbot.generate_response(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")


def initialize_sample_documents(rag_agent):
    """Initialize RAG agent with sample documents for testing"""
    sample_docs = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. "
            "It was created by Guido van Rossum and first released in 1991.",
            "metadata": {"source": "python_intro", "type": "programming"},
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn "
            "and improve from experience without being explicitly programmed.",
            "metadata": {"source": "ml_basics", "type": "ai"},
        },
        {
            "content": "ChromaDB is an open-source vector database designed for AI applications. "
            "It provides efficient storage and retrieval of embeddings.",
            "metadata": {"source": "chromadb_info", "type": "database"},
        },
        {
            "content": "Ollama is a tool that allows you to run large language models locally on your machine. "
            "It supports various models like Llama, Mistral, and others.",
            "metadata": {"source": "ollama_info", "type": "tools"},
        },
    ]

    print("Initializing with sample documents...")
    for i, doc in enumerate(sample_docs):
        rag_agent.add_document(doc["content"], doc["metadata"], doc_id=f"sample_{i}")
        print(f"  Added: {doc['metadata']['source']}")


def test_rag_agent_standalone():
    """Test the RAG agent functionality independently"""
    if not RAG_AVAILABLE:
        print("RAG dependencies not available, skipping RAG agent tests")
        return

    print("\n" + "=" * 60)
    print("Testing RAG Agent Standalone Functionality")
    print("=" * 60)

    # Initialize RAG agent
    agent = RAGAgent()

    # Test 1: Add sample documents
    print("\n1. Adding Sample Documents:")
    print("-" * 30)
    sample_docs = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. "
            "It was created by Guido van Rossum and first released in 1991.",
            "metadata": {"source": "python_intro", "type": "programming"},
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn "
            "and improve from experience without being explicitly programmed.",
            "metadata": {"source": "ml_basics", "type": "ai"},
        },
        {
            "content": "ChromaDB is an open-source vector database designed for AI applications. "
            "It provides efficient storage and retrieval of embeddings.",
            "metadata": {"source": "chromadb_info", "type": "database"},
        },
        {
            "content": "Ollama is a tool that allows you to run large language models locally on your machine. "
            "It supports various models like Llama, Mistral, and others.",
            "metadata": {"source": "ollama_info", "type": "tools"},
        },
    ]

    for i, doc in enumerate(sample_docs):
        doc_id = agent.add_document(
            doc["content"], doc["metadata"], doc_id=f"sample_{i}"
        )
        print(f"✓ Added: {doc['metadata']['source']} (ID: {doc_id})")

    # Test 2: Collection stats
    print("\n2. Collection Statistics:")
    print("-" * 30)
    stats = agent.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test 3: List documents
    print("\n3. Document Listing:")
    print("-" * 30)
    docs = agent.list_documents()
    for i, doc in enumerate(docs, 1):
        source = doc["metadata"].get("source", "unknown")
        print(f"  {i}. ID: {doc['id']}, Source: {source}")
        print(f"     Preview: {doc['content'][:80]}...")

    # Test 4: Document retrieval
    print("\n4. Document Retrieval Tests:")
    print("-" * 30)
    test_queries = [
        "What is Python?",
        "Tell me about machine learning",
        "What is ChromaDB?",
        "How does Ollama work?",
        "What is the capital of France?",  # Not in context
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        docs = agent.retrieve_documents(query, top_k=2)
        print(f"Retrieved {len(docs)} relevant documents:")
        for i, doc in enumerate(docs, 1):
            source = doc["metadata"].get("source", "unknown")
            distance = doc["distance"]
            print(
                f"  {i}. {source} (similarity: {1-distance:.3f}, distance: {distance:.3f})"
            )

    # Test 5: RAG response generation
    print("\n5. RAG Response Generation:")
    print("-" * 30)
    for query in test_queries[:3]:  # Test first 3 queries
        print(f"\nQuery: {query}")
        response = agent.generate_response(query)
        print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")

    # Test 6: Document management
    print("\n6. Document Management:")
    print("-" * 30)

    # Add a test document
    test_content = "The quick brown fox jumps over the lazy dog. This is a test document for deletion."
    test_doc_id = agent.add_document(
        test_content, {"source": "test_deletion", "type": "test"}
    )
    print(f"✓ Added test document: {test_doc_id}")

    # Verify it exists
    docs_before = len(agent.list_documents())
    print(f"Documents before deletion: {docs_before}")

    # Delete the document
    success = agent.delete_document(test_doc_id)
    print(f"✓ Deleted document: {success}")

    # Verify deletion
    docs_after = len(agent.list_documents())
    print(f"Documents after deletion: {docs_after}")
    print("\n✓ RAG Agent standalone test completed!")


def test_rag_chatbot():
    """Test the RAG chatbot functionality"""

    print("\nTesting SimpleChatbot (RAG Mode)...")

    chatbot = SimpleChatbot(use_function_agent=False, use_rag_agent=True)

    if chatbot.rag_agent:
        # Initialize with sample documents
        initialize_sample_documents(chatbot.rag_agent)

        # Test adding a custom document
        test_doc = "The quick brown fox jumps over the lazy dog. This is a test document for RAG functionality."
        doc_id = chatbot.rag_agent.add_document(
            test_doc, {"source": "test_doc", "type": "test"}
        )
        print(f"Added test document with ID: {doc_id}")

    test_queries = [
        "What is Python?",  # Should use existing context
        "Tell me about the fox",  # Should use our test document
        "What's machine learning?",  # Should use existing context
        "What color is the fox?",  # Should reference our test document
    ]

    for query in test_queries:
        print("=======================================================")
        print(f"\nRAG Query: {query}")
        try:
            response = chatbot.generate_response(query)
            print(f"Response: {response}")

            # Show retrieved documents
            if chatbot.rag_agent:
                docs = chatbot.rag_agent.retrieve_documents(query, top_k=2)
                print(f"Retrieved {len(docs)} relevant documents:")
                for i, doc in enumerate(docs, 1):
                    source = doc["metadata"].get("source", "unknown")
                    print(f"  {i}. {source} (distance: {doc['distance']:.3f})")

        except Exception as e:
            print(f"Error: {e}")


def test_combined_chatbot():
    """Test chatbot with both function calling and RAG"""
    print("\nTesting SimpleChatbot (Combined Mode)...")

    chatbot = SimpleChatbot(use_function_agent=True, use_rag_agent=True)

    if chatbot.rag_agent:
        # Initialize with sample documents for combined testing
        initialize_sample_documents(chatbot.rag_agent)

    test_queries = [
        "What time is it?",  # Should use function calling
        "What is Python?",  # Should use RAG
        "Calculate 15 * 8",  # Should use function calling
        "Tell me about ChromaDB",  # Should use RAG
    ]

    for query in test_queries:
        print("=======================================================")
        print(f"\nCombined Query: {query}")
        try:
            response = chatbot.generate_response(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
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
