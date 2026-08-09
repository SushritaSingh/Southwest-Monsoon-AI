# scripts/test_paper_tool.py
from agents.paper_tool import query_system_paper

if __name__ == "__main__":
    print("Testing Paper Query Tool...\n")
    
    # Test Query 1
    response1 = query_system_paper.invoke({"query": "What models are used in Part 01?"})
    print("--- Query 1 Result ---")
    print(response1)
    
    print("\n" + "="*40 + "\n")
    
    # Test Query 2
    response2 = query_system_paper.invoke({"query": "What is the primary objective of the platform?"})
    print("--- Query 2 Result ---")
    print(response2)