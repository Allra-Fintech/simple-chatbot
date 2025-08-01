#!/usr/bin/env python3

import ollama

try:
    from .function_agent import FunctionCallerAgent
except ImportError:
    FunctionCallerAgent = None

try:
    from .rag_agent import RAGAgent
except ImportError:
    RAGAgent = None


class SimpleChatbot:
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        use_function_agent: bool = True,
        use_rag_agent: bool = False,
    ):
        self.model_name = model_name

        # Initialize function calling agent
        self.use_function_agent = use_function_agent
        if use_function_agent and FunctionCallerAgent:
            self.function_agent = FunctionCallerAgent(model_name)
        else:
            self.function_agent = None

        # Initialize RAG agent
        self.use_rag_agent = use_rag_agent
        if use_rag_agent and RAGAgent:
            self.rag_agent = RAGAgent(model_name)
        else:
            self.rag_agent = None

    def generate_response(self, user_query: str) -> str:
        """Generate response with optional function calling and RAG"""

        # Use RAG agent if enabled
        if self.use_rag_agent and self.rag_agent:
            try:
                return self.rag_agent.generate_response(user_query)
            except Exception as e:
                print(
                    f"RAG agent failed: {e}, falling back to function agent or regular chat"
                )

        # Use function calling agent if enabled
        if self.use_function_agent and self.function_agent:
            try:
                return self.function_agent.invoke(user_query)
            except Exception as e:
                print(f"Function agent failed: {e}, using regular chat")

        # Fallback to regular chat
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": user_query}],
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def chat_loop(self):
        """Interactive chat loop"""
        # Build status message
        features = []
        if self.use_rag_agent:
            features.append("RAG")
        if self.use_function_agent:
            features.append("function calling")

        if features:
            agent_status = f"with {' and '.join(features)}"
        else:
            agent_status = "in basic mode"

        print(f"Simple Chatbot initialized {agent_status}!")
        print("Commands: 'quit' to exit")

        if self.use_rag_agent and self.rag_agent:
            print(
                "RAG commands: 'add_doc <text>', 'add_file <path>', 'list_docs', 'clear_docs'"
            )

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            elif (
                self.use_rag_agent
                and self.rag_agent
                and self._handle_rag_commands(user_input)
            ):
                continue
            else:
                response = self.generate_response(user_input)
                print(f"Bot: {response}")

    def _handle_rag_commands(self, user_input: str) -> bool:
        """Handle RAG-specific commands. Returns True if command was handled."""
        if not (self.use_rag_agent and self.rag_agent):
            return False

        if user_input.startswith("add_doc "):
            doc_text = user_input[8:].strip()
            if doc_text:
                try:
                    doc_id = self.rag_agent.add_document(doc_text)
                    print(f"✓ Document added with ID: {doc_id}")
                except Exception as e:
                    print(f"✗ Error adding document: {e}")
            else:
                print("Usage: add_doc <document text>")
            return True

        elif user_input.startswith("add_file "):
            file_path = user_input[9:].strip()
            if file_path:
                try:
                    doc_id = self.rag_agent.add_document_from_file(file_path)
                    print(f"✓ File added with ID: {doc_id}")
                except Exception as e:
                    print(f"✗ Error adding file: {e}")
            else:
                print("Usage: add_file <file path>")
            return True

        elif user_input == "list_docs":
            docs = self.rag_agent.list_documents()
            if docs:
                print(f"\n📚 Documents in collection ({len(docs)} total):")
                for i, doc in enumerate(docs, 1):
                    source = doc["metadata"].get("source", "unknown")
                    print(f"  {i}. ID: {doc['id']}, Source: {source}")
                    print(f"     Preview: {doc['content'][:100]}...")
            else:
                print("No documents in collection")
            return True

        elif user_input == "clear_docs":
            if self.rag_agent.clear_collection():
                print("✓ Document collection cleared")
            else:
                print("✗ Error clearing collection")
            return True

        return False


def main():
    print("Initializing Simple Chatbot with Ollama...")

    # Check if Ollama is running
    try:
        ollama.list()
    except Exception:
        print("Error: Could not connect to Ollama. Make sure Ollama is running.")
        print("Install Ollama from https://ollama.ai and run: ollama serve")
        return

    # Ask user for RAG preference
    use_rag = False
    if RAGAgent:
        print("\nWould you like to enable RAG (Retrieval Augmented Generation)? (y/n)")
        rag_choice = input().strip().lower()
        use_rag = rag_choice in ["y", "yes"]
    elif input(
        "\nRAG dependencies not available. Continue with basic chatbot? (y/n): "
    ).strip().lower() not in ["y", "yes"]:
        return

    chatbot = SimpleChatbot(use_rag_agent=use_rag)
    chatbot.chat_loop()


if __name__ == "__main__":
    main()
