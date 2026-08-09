# rag/retriever.py
"""
Prompt Context Retriever Module.
Extracts matches from the Vector DB and wraps them with LLM instruction sets.
"""

from typing import Dict, Any, List
from rag.vector_store import MonsoonVectorDB

class MonsoonRetriever:
    """Orchestrates query translation, context matching, and prompt template injection."""
    
    def __init__(self):
        self.db = MonsoonVectorDB()

    def get_grounded_prompt(self, user_query: str) -> Dict[str, Any]:
        """Retrieves matching documents and formats the final prompt."""
        matches = self.db.search(user_query, top_k=2)
        
        context_str = ""
        if matches:
            context_str = "\n\n".join([f"Source: {m['source']}\nContent: {m['context']}" for m in matches])
        else:
            context_str = "No active scientific context files are cataloged for this inquiry."
            
        prompt_template = (
            "You are an expert meteorological system. Use the scientific sources listed below to answer "
            "the user's questions clearly, factually, and without speculation.\n\n"
            f"--- Scientific Sources ---\n{context_str}\n\n"
            f"User Query: {user_query}\n"
            "Formulate your grounded scientific assessment:"
        )
        
        return {
            "prompt": prompt_template,
            "references": matches
        }