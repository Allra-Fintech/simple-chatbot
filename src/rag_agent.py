#!/usr/bin/env python3

from typing import List, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama


class RAGAgent:
    """RAG (Retrieval Augmented Generation) Agent with ChromaDB and Ollama"""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents",
        top_k: int = 3,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.collection_name = collection_name

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document collection for RAG"},
        )

    def add_document(
        self,
        content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a document to the vector database"""
        if not content.strip():
            raise ValueError("Document content cannot be empty")

        if doc_id is None:
            doc_id = f"doc_{self.collection.count()}"

        if metadata is None:
            metadata = {}

        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()

        # Add to collection
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id],
        )

        return doc_id

    def add_document_from_file(
        self, file_path: str, metadata: Optional[dict] = None
    ) -> str:
        """Add a document from a file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        # Add source information to metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix,
            }
        )

        doc_id = f"file_{file_path.stem}_{self.collection.count()}"
        return self.add_document(content, metadata, doc_id)

    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Retrieve relevant documents for a query"""
        if top_k is None:
            top_k = self.top_k

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append(
                    {
                        "content": doc,
                        "metadata": (
                            results["metadatas"][0][i]
                            if results["metadatas"][0]
                            else {}
                        ),
                        "distance": (
                            results["distances"][0][i]
                            if results["distances"][0]
                            else 0.0
                        ),
                        "id": results["ids"][0][i] if results["ids"][0] else f"doc_{i}",
                    }
                )

        return documents

    def generate_response(self, query: str, include_context: bool = True) -> str:
        """Generate response using RAG approach"""
        try:
            context_text = ""

            if include_context:
                # Retrieve relevant documents
                relevant_docs = self.retrieve_documents(query)

                if relevant_docs:
                    context_parts = []
                    for doc in relevant_docs:
                        source = doc["metadata"].get("source", "unknown")
                        context_parts.append(
                            f"Source: {source}\nContent: {doc['content']}"
                        )

                    context_text = "\n\n".join(context_parts)

            # Create prompt with context
            if context_text:
                prompt = f"""Based on the following context information, please answer the question."
                "If the context doesn't contain relevant information, you can use your general knowledge"
                "but mention that the information is not from the provided context.

Context:
{context_text}

Question: {query}

Answer:"""
            else:
                prompt = query

            # Generate response using Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )

            return response["message"]["content"]

        except Exception as e:
            return f"Error generating RAG response: {str(e)}"

    def list_documents(self) -> List[dict]:
        """List all documents in the collection"""
        try:
            # Get all documents
            results = self.collection.get(include=["documents", "metadatas"])

            documents = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    documents.append(
                        {
                            "id": results["ids"][i] if results["ids"] else f"doc_{i}",
                            "content": doc[:200] + "..." if len(doc) > 200 else doc,
                            "metadata": (
                                results["metadatas"][i] if results["metadatas"] else {}
                            ),
                        }
                    )

            return documents
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document collection for RAG"},
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False

    def get_collection_stats(self) -> dict:
        """Get statistics about the document collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            }
        except Exception as e:
            return {"error": str(e)}
