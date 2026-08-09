# agents/paper_tool.py
import chromadb

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool


@tool
def query_system_paper(query: str) -> str:
    """Queries the Weather Intelligence System research paper to retrieve methodology details, math formulas, and findings."""
    try:
        chroma_client = chromadb.PersistentClient(path="./rag/chroma_db")
        collection = chroma_client.get_collection(name="weather_paper_db")

        results = collection.query(query_texts=[query], n_results=2)

        if results["documents"] and results["documents"][0]:
            context = "\n---\n".join(results["documents"][0])
            return f"Citation [Research Paper Draft]:\n{context}"
        return "No relevant sections found in the research paper draft."
    except Exception as e:
        return f"Could not access paper database: {str(e)}"