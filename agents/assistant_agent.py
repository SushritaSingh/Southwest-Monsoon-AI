# agents/assistant_agent.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.paper_tool import query_system_paper

def initialize_agent():
    tools = [query_system_paper]
    print("✅ Successfully initialized Weather Assistant Agent with paper retrieval tools!")
    return tools

if __name__ == "__main__":
    initialize_agent()