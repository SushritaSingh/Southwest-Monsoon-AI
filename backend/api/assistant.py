# backend/api/assistant.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from agents.workflows import monsoon_agent_app  # Imported from your agents module initialization

router = APIRouter(prefix="/assistant", tags=["Generative AI Agent"])

class InquiryPayload(BaseModel):
    query: str

@router.post("/query")
async def ask_agentic_assistant(payload: InquiryPayload) -> Dict[str, Any]:
    """
    Submits a user prompt to the compiled multi-agent LangGraph system for analysis.
    """
    try:
        initial_state = {
            "messages": [HumanMessage(content=payload.query)],
            "next_action": "",
            "grounded_context": "",
            "computed_forecast": ""
        }
        
        # Invoke compiled agent workflow graph
        result = monsoon_agent_app.invoke(initial_state)
        
        # Extract output messages
        steps: List[str] = [msg.content for msg in result.get("messages", [])]
        
        return {
            "agent_responses": steps,
            "grounded_context": result.get("grounded_context", ""),
            "computed_forecast": result.get("computed_forecast", ""),
            "status": "Completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow execution failed: {str(e)}")