#!/usr/bin/env python3

from datetime import datetime
from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    """
    try:
        # Simple whitelist of allowed characters for security
        allowed_chars = set("0123456789+-*/().\\s")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


class FunctionCallerAgent:
    """LangGraph-based function calling agent"""

    def __init__(self, model_name):
        # Use the provided model name
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name, temperature=0)

        # Create tools list
        self.tools = [get_current_time, calculator]

        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create the graph
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""

        def should_continue(state: MessagesState) -> str:
            """Determine if we should continue or end the conversation"""
            last_message = state["messages"][-1]

            # If there are tool calls, continue to tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # Otherwise, end
            return END

        def call_model(state: MessagesState) -> Dict[str, Any]:
            """Call the language model with tools"""
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # Create the graph
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def invoke(self, message: str) -> str:
        """Process a message through the function calling agent"""
        try:
            # Create the input state
            initial_state = {
                "messages": [
                    SystemMessage(
                        content="""You are a helpful AI assistant with access to tools.
                    Use the available tools when appropriate to answer questions accurately.
                    For calculations, use the calculator tool.
                    For current time, use the get_current_time tool."""
                    ),
                    HumanMessage(content=message),
                ]
            }

            # Run the graph
            result = self.graph.invoke(initial_state)

            # Extract the final response
            final_message = result["messages"][-1]
            return final_message.content

        except Exception as e:
            return f"Error processing request: {str(e)}"

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]


def test_function_agent():
    """Test the function calling agent"""
    agent = FunctionCallerAgent()

    test_queries = [
        "What time is it?",
        "Calculate 15 * 7 + 23",
        "What's the square root of 144?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        response = agent.invoke(query)
        print(f"Response: {response}")


if __name__ == "__main__":
    test_function_agent()
