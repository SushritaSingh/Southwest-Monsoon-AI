# rag/latex_ingest.py
import re
import chromadb
from rag.embed_pipeline import EmbedPipeline


def clean_latex_source(tex_content: str) -> str:
    """Strips LaTeX formatting commands so clean text gets stored in vector DB."""
    # Remove section titles and extract plain text
    text = re.sub(
        r"\\(?:section|subsection|subsubsection)\*?\{([^}]+)\}", r"\1", tex_content
    )
    # Remove other standard backslash commands
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*", "", text)
    # Remove LaTeX comments
    text = re.sub(r"%.*", "", text)
    # Clean up excess whitespaces
    return re.sub(r"\s+", " ", text).strip()


def ingest_paper_to_vector_db(tex_file_path: str):
    """Chunks the paper text, generates embeddings, and indexes them into ChromaDB."""
    pipeline = EmbedPipeline(chunk_size=300, chunk_overlap=30)

    try:
        raw_tex = pipeline.read_text(tex_file_path)
    except FileNotFoundError:
        print(f"Error: Could not find paper file at '{tex_file_path}'.")
        print("Please check the path to your LaTeX file and try again.")
        return

    clean_text = clean_latex_source(raw_tex)

    if not clean_text:
        print("Warning: The target LaTeX file appears to be empty.")
        return

    chunks = pipeline.chunk_text(clean_text)
    embeddings = pipeline.generate_embeddings(chunks)

    # Initialize ChromaDB persistent client
    chroma_client = chromadb.PersistentClient(path="./rag/chroma_db")
    collection = chroma_client.get_or_create_collection(name="weather_paper_db")

    # Store chunks into vector store
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": "system_research_paper"} for _ in chunks],
        ids=[f"paper_chunk_{i}" for i in range(len(chunks))],
    )

    print(
        f"Successfully ingested {len(chunks)} paper chunks into ChromaDB ('weather_paper_db')."
    )


if __name__ == "__main__":
    # Point this to your full Overleaf LaTeX paper file
    ingest_paper_to_vector_db("paper/main.tex")