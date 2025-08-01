# RAG and Function Calling Demo

A simple educational project showing how **RAG (Retrieval Augmented Generation)** and **Function Calling** work with LLMs.

## What This Demonstrates

### RAG (Retrieval Augmented Generation)
- How to convert documents into vectors (embeddings)
- How to find relevant documents using similarity search
- How to use retrieved documents as context for better answers

### Function Calling
- How LLMs can call external tools/functions
- Simple examples: time lookup and calculations
- How to route queries to appropriate functions

## Quick Start

1. **Install Ollama**:
   ```bash
   # Visit https://ollama.ai and install
   ollama serve
   ollama pull llama3.2:3b
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**:
   ```bash
   python chatbot.py
   ```

## Demo Flow

The demo shows both concepts in action:

### Function Calling Demo
```
User: What time is it?
Bot: The current time is 2024-01-15 14:30:25

User: Calculate 15 * 7 + 23
Bot: 15 * 7 + 23 = 128
```

### RAG Demo
```
Adding documents: "Python is a programming language...", "Machine learning..."

User: Who created Python?
Bot: Based on the documents, Python was created by Guido van Rossum in 1991.
```

## Code Structure

- **`src/main.py`** - Simple demo runner
- **`src/function_agent.py`** - Function calling logic (50 lines)
- **`src/rag_agent.py`** - RAG implementation (100 lines)
- **`test_demo.py`** - Basic tests

## Key Concepts Shown

### RAG Process:
1. Convert documents to embeddings (vectors)
2. Convert user query to embedding
3. Find most similar documents (cosine similarity)
4. Use relevant documents as context in LLM prompt

### Function Calling Process:
1. Analyze user query for function needs
2. Extract parameters (e.g., math expression)
3. Call appropriate function
4. Format result into natural response

## Educational Value

This project strips away complexity to show the **core concepts**:
- No complex frameworks (minimal dependencies)
- Clear, readable code with comments
- Step-by-step process demonstration
- Easy to modify and experiment with

## Requirements

- Python 3.7+
- Ollama running locally
- Dependencies: `ollama`, `sentence-transformers`, `numpy`

Perfect for learning how modern AI applications work under the hood!