#!/usr/bin/env python3
"""
Main entry point for the Simple Chatbot application.

This script provides a command-line interface to run the chatbot with
various options including function calling and RAG capabilities.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import main

if __name__ == "__main__":
    main()