# pages/7_AI_Weather_Assistant.py
import os
import streamlit as st
import numpy as np
from langchain_core.messages import HumanMessage
from agents.workflows import monsoon_agent_app
from rag.vector_store import MonsoonVectorDB

st.set_page_config(page_title="Agentic Research AI Assistant", layout="wide")

st.title("💬 Agentic Multi-Agent Weather Research Assistant")
st.markdown("---")

st.info("💡 This view orchestrates collaborative multi-agent runs using LangGraph and FAISS Vector Search.")

# Initialize Vector DB instance
db = MonsoonVectorDB()

# Ingestion Section
st.subheader("1. Knowledge Base Document Ingestion")
uploaded_research_pdf = st.file_uploader("Upload Meteorological Reference Document (PDF)", type="pdf")

if uploaded_research_pdf:
    os.makedirs("data", exist_ok=True)
    temp_destination = f"data/{uploaded_research_pdf.name}"
    with open(temp_destination, "wb") as f:
        f.write(uploaded_research_pdf.getbuffer())
    with st.spinner("Analyzing document structure and creating vector embeddings..."):
        db.populate_index(temp_destination)
    st.success("✅ Knowledge Base successfully updated and stored in FAISS Index.")

# Query Section & Quick Prompts
st.subheader("2. Operational Research Query")

st.markdown("**Quick Preset Queries:**")
col1, col2, col3 = st.columns(3)
selected_prompt = None

if col1.button("🌀 Cyclone Genesis Indicators", use_container_width=True):
    selected_prompt = "What are the primary thermodynamic and vorticity indicators required for tropical cyclone genesis?"
if col2.button("🌧️ Monsoonal Onset Anomalies", use_container_width=True):
    selected_prompt = "How do Sea Surface Temperatures (SST) over the Arabian Sea influence the southwest monsoon onset timing?"
if col3.button("📊 Forecast Index Interpretation", use_container_width=True):
    selected_prompt = "Explain how NDVI and NDWI indices correlate with post-monsoon crop yield and flood risk."

user_query = st.text_input(
    "Enter your research or operational forecasting question:", 
    value=selected_prompt if selected_prompt else ""
)

if user_query:
    st.markdown("---")
    
    # LangGraph Execution Flow Display
    with st.status("🤖 Multi-Agent Graph Orchestration Active...", expanded=True) as status:
        st.write("🔍 **Retriever Agent:** Querying FAISS embeddings for grounded context...")
        
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "next_action": "",
            "grounded_context": "",
            "computed_forecast": ""
        }
        
        # Invoke LangGraph app workflow
        result = monsoon_agent_app.invoke(initial_state)
        
        st.write("🧠 **Reasoning Agent:** Evaluating atmospheric physical parameters...")
        st.write("📝 **Synthesis Agent:** Formatting structured scientific diagnostic report...")
        status.update(label="✅ Agentic Execution Complete!", state="complete", expanded=False)

    # Multi-Agent Output Display
    st.markdown("### 📋 Multi-Agent Diagnostic Steps & Output")
    
    # Render messages inside chat containers for clean UI
    for msg in result.get("messages", []):
        role = "user" if getattr(msg, "type", "") == "human" or isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # Grounded Metadata Analysis Section
    st.markdown("---")
    st.markdown("### 🔬 Extracted Grounding & Diagnostic Metadata")
    
    meta_col1, meta_col2 = st.columns([3, 1])
    
    with meta_col1:
        with st.expander("📄 View Retrieved Context Snippets", expanded=False):
            st.code(result.get("grounded_context", "No context found"), language="markdown")
            
    with meta_col2:
        st.metric("Forecast Values Interrogated", str(result.get("computed_forecast", "N/A")))