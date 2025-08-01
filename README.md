# Simple Chatbot with Function Calling and RAG

A Python-based chatbot that uses Ollama for local LLM inference with optional function calling and RAG (Retrieval Augmented Generation) capabilities.

## Features

- **Local LLM Integration**: Uses Ollama for private, local language model inference
- **Function Calling**: Optional function calling with built-in tools:
  - Current time retrieval
  - Mathematical calculations
- **RAG (Retrieval Augmented Generation)**: Optional document-based knowledge retrieval:
  - Vector-based document storage with ChromaDB
  - Semantic search using sentence transformers
  - Interactive document management commands
  - Persistent document storage
- **Interactive Chat**: Command-line interface for real-time conversations
- **Flexible Modes**: Choose between basic chat, function calling, RAG, or combined modes
- **Fallback Support**: Gracefully falls back through modes if one fails

## Prerequisites

Before installation, you need to have Ollama installed and running:

1. **Install Ollama**:
   - Visit [https://ollama.ai](https://ollama.ai) and download Ollama for your platform
   - Follow the installation instructions for your operating system

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull a language model**:
   ```bash
   ollama pull llama3.2:3b
   # or any other model you prefer
   ```

## Installation

### Option 1: Automated Setup (Recommended)

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd simple-chatbot
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```
   
   This will automatically:
   - Install all required Python dependencies
   - Verify that Ollama is running and accessible
   - Confirm the setup is complete

### Option 2: Manual Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Ollama is running**:
   ```bash
   # Test that Ollama is accessible
   ollama list
   ```

## Usage

### Start the Chatbot

```bash
python chatbot.py
```

Or run from the source module:
```bash
python -m src.main
```

When you start the chatbot, you'll be asked if you want to enable RAG functionality:

```
Initializing Simple Chatbot with Ollama...

Would you like to enable RAG (Retrieval Augmented Generation)? (y/n)
```

### Chat Modes

The chatbot can operate in different modes:

- **Basic Mode**: Simple chat without function calling or RAG
- **Function Calling Mode**: Includes time and calculation tools
- **RAG Mode**: Document-based knowledge retrieval
- **Combined Mode**: Both function calling and RAG enabled

### Example Interactions

#### Basic Function Calling
```
You: What time is it?
Bot: The current date and time is 2024-01-15 14:30:25

You: Calculate 25 * 4 + 10
Bot: 110
```

#### RAG Interactions
```
You: What is Python?
Bot: Based on the context information, Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

You: Tell me about ChromaDB
Bot: ChromaDB is an open-source vector database designed for AI applications. It provides efficient storage and retrieval of embeddings.
```

#### RAG Commands
When RAG is enabled, you have access to these commands:

```
add_doc <text>              # Add a document from text
add_file <path>             # Add a document from file
list_docs                   # List all documents
clear_docs                  # Clear all documents
quit                        # Exit the chatbot
```

#### Example RAG Commands
```
You: add_doc Machine learning is a subset of AI that enables computers to learn from data.
Bot: ✓ Document added with ID: doc_4

You: list_docs
Bot: 📚 Documents in collection (5 total):
  1. ID: sample_0, Source: python_intro
     Preview: Python is a high-level programming language known for its simplicity and readability...
  2. ID: sample_1, Source: ml_basics
     Preview: Machine learning is a subset of artificial intelligence that enables computers to learn...

You: add_file ./my_document.txt
Bot: ✓ File added with ID: file_my_document_5
```

### Testing

Run the test script to verify functionality:

```bash
python run_tests.py
```

Or run tests directly:
```bash
python tests/test_simple.py
```

## Configuration

### Changing the LLM Model

You can modify the model used by editing the `SimpleChatbot` initialization in `main.py`:

```python
chatbot = SimpleChatbot(model_name="llama2:latest")  # or any other model
```

### Disabling Function Calling

To run the chatbot without function calling capabilities:

```python
chatbot = SimpleChatbot(use_function_agent=False)
```

## Available Functions

When function calling is enabled, the chatbot has access to:

- **get_current_time()**: Returns the current date and time
- **calculator(expression)**: Safely evaluates mathematical expressions

## Dependencies

### Core Dependencies
- `ollama`: Ollama Python client
- `langchain`: LangChain framework
- `langchain-community`: Additional LangChain components
- `langgraph`: Graph-based workflow management
- `langchain-ollama`: Ollama integration for LangChain

### RAG Dependencies
- `chromadb`: Vector database for document storage
- `sentence-transformers`: Text embedding generation

## Troubleshooting

### "Could not connect to Ollama"
- Ensure Ollama is installed and running (`ollama serve`)
- Check that you have pulled at least one model (`ollama pull llama3.2:3b`)

### "Function agent failed"
- The chatbot will automatically fall back to basic chat mode
- Check that all dependencies are installed correctly

### Import Errors
- Run `python setup.py` to reinstall dependencies
- Ensure you're using Python 3.7 or later

## Project Structure

```
simple-chatbot/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── main.py            # Main chatbot class and CLI
│   ├── function_agent.py  # Function calling agent
│   └── rag_agent.py       # RAG implementation
├── tests/                 # Test suite
│   ├── __init__.py        # Test package initialization
│   └── test_simple.py     # Comprehensive tests
├── chatbot.py             # Main entry point
├── run_tests.py           # Test runner
├── setup.py               # Setup and installation script
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── CLAUDE.md              # Development instructions
```

## Architecture

- **SimpleChatbot**: Main chatbot class with optional function calling and RAG
- **FunctionCallerAgent**: LangGraph-based agent for tool integration
- **RAGAgent**: ChromaDB-based document retrieval and generation system
- **Tools**: Modular functions for time and calculations
- **Vector Database**: ChromaDB with sentence-transformers for embeddings
- **Fallback System**: Graceful degradation through RAG → Function Calling → Basic Chat

### RAG Architecture

The RAG system includes:
- **Document Ingestion**: Text and file-based document addition
- **Vector Storage**: ChromaDB persistent storage with metadata
- **Embedding Generation**: sentence-transformers for semantic search
- **Context Retrieval**: Top-k document retrieval based on query similarity
- **Response Generation**: Context-aware responses using retrieved documents

## License

This project is open source. Please check the repository for license details.