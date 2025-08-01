#!/usr/bin/env python3
"""
Simple function calling demo - shows how LLM can call external tools
"""

from datetime import datetime
import ollama


def get_current_time() -> str:
    """Get the current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely"""
    try:
        # Simple whitelist for security
        allowed_chars = set("0123456789+-*/().\\s")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters"

        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def simple_function_call(query: str, model: str = "llama3.2:3b") -> str:
    """
    Simple function calling demo - no complex framework needed
    Shows the core concept: LLM decides when to call functions
    """

    # Check if query needs time
    if "time" in query.lower():
        time_result = get_current_time()

        # Ask LLM to format the response nicely
        prompt = f"The user asked: '{query}'. The current time is: {time_result}. Please provide a natural response."
        response = ollama.chat(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    # Check if query needs calculation
    elif any(
        word in query.lower() for word in ["calculate", "math", "*", "+", "-", "/", "="]
    ):
        # Extract calculation (simple approach)
        import re

        # Find mathematical expressions
        math_pattern = r"[0-9+\-*/\s()]+"
        match = re.search(math_pattern, query)

        if match:
            expression = match.group().strip()
            calc_result = calculator(expression)

            # Ask LLM to format the response
            prompt = f"The user asked: '{query}'. The calculation '{expression}' equals: {calc_result}."
            " Please provide a natural response."
            response = ollama.chat(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]

    # Default: regular chat
    response = ollama.chat(model=model, messages=[{"role": "user", "content": query}])
    return response["message"]["content"]


if __name__ == "__main__":
    # Simple test
    test_queries = ["What time is it?", "Calculate 15 * 7 + 23", "Hello, how are you?"]

    for query in test_queries:
        print(f"\nUser: {query}")
        response = simple_function_call(query)
        print(f"Bot: {response}")
