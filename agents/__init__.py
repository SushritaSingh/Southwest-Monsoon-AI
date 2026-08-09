# agents/__init__.py
"""
Agents module initialization for the Weather Intelligence and Climate Decision Support Platform.
Exposes the compiled multi-agent LangGraph workflow and associated tools.
"""

from agents.workflows import monsoon_agent_app
from agents.tools import (
    fetch_live_weather,
    run_predictive_forecast,
    search_scientific_knowledge_base
)

# Expose key components at the package level
__all__ = [
    "monsoon_agent_app",
    "fetch_live_weather",
    "run_predictive_forecast",
    "search_scientific_knowledge_base"
]