#!/usr/bin/env python3
"""
Simple RAG demo - shows how to retrieve relevant documents and use them for generation
"""

from sentence_transformers import SentenceTransformer
import ollama
import numpy as np


def create_embeddings(texts: list, model_name: str = "all-MiniLM-L6-v2"):
    """Create embeddings for a list of texts"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings, model


def find_most_relevant(query: str, documents: list, embeddings, model, top_k: int = 2):
    """Find most relevant documents using cosine similarity"""
    # Get query embedding
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarities = np.dot(query_embedding, embeddings.T).flatten()

    # Get top-k most similar documents
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    relevant_docs = []
    for idx in top_indices:
        relevant_docs.append(
            {"content": documents[idx], "similarity": similarities[idx]}
        )

    return relevant_docs


def simple_rag_query(query: str, documents: list, model_name: str = "llama3.2:3b"):
    """
    Simple RAG demo - core concept in minimal code
    1. Convert documents to embeddings (vectors)
    2. Find most similar documents to the query
    3. Use those documents as context for the LLM
    """

    if not documents:
        # No documents, just use regular chat
        response = ollama.chat(
            model=model_name, messages=[{"role": "user", "content": query}]
        )
        return response["message"]["content"]

    # Step 1: Create embeddings for all documents
    embeddings, embedding_model = create_embeddings(documents)

    # Step 2: Find most relevant documents
    relevant_docs = find_most_relevant(query, documents, embeddings, embedding_model)

    # Step 3: Create context from relevant documents
    context = "\n\n".join([doc["content"] for doc in relevant_docs])

    # Step 4: Ask LLM with context
    prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model=model_name, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    # Simple test
    test_docs = [
        "Python is a programming language created by Guido van Rossum in 1991.",
        "Machine learning is a subset of AI that learns from data.",
        "ChromaDB is a vector database for AI applications.",
    ]

    test_queries = [
        "Who created Python?",
        "What is machine learning?",
        "Tell me about ChromaDB",
    ]

    print("=== RAG Demo ===")
    for query in test_queries:
        print(f"\nUser: {query}")
        response = simple_rag_query(query, test_docs)
        print(f"Bot: {response}")
