#!/usr/bin/env python3

import subprocess
import sys


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False


def check_ollama():
    """Check if Ollama is available"""
    try:
        import ollama

        ollama.list()
        print("✓ Ollama is running and accessible")
        return True
    except ImportError:
        print("✗ Ollama package not installed")
        return False
    except Exception:
        print("✗ Ollama is not running. Please start Ollama service:")
        print("  1. Install Ollama from https://ollama.ai")
        print("  2. Run: ollama serve")
        print("  3. Pull a model: ollama pull llama2:latest")
        return False


def check_rag_dependencies():
    """Check if RAG dependencies are available"""
    try:
        import chromadb
        import sentence_transformers
        print("✓ RAG dependencies (ChromaDB and sentence-transformers) available")
        return True
    except ImportError as e:
        print(f"✗ RAG dependencies missing: {e}")
        print("  This will be resolved by installing requirements.txt")
        return False


def main():
    print("Simple Chatbot with RAG Setup")
    print("============================")

    # Install requirements
    if not install_requirements():
        return

    # Check Ollama
    if not check_ollama():
        print("\nSetup incomplete. Please install and start Ollama, then try again.")
        return
    
    # Check RAG dependencies
    check_rag_dependencies()

    print("\n✓ Setup complete! You can now run the chatbot with:")
    print("  python chatbot.py")
    print("\nThe chatbot includes:")
    print("  - Function calling (time, calculations)")
    print("  - RAG (document retrieval with ChromaDB)")
    print("  - Interactive document management")
    
    print("\nOther commands:")
    print("  python run_tests.py     # Run all tests")
    print("  python -m src.main      # Run from src module")


if __name__ == "__main__":
    main()
