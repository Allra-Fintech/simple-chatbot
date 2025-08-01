#!/usr/bin/env python3
"""
Simple entry point for RAG and Function Calling demo
"""

import sys
import os
from src.main import main

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    main()
