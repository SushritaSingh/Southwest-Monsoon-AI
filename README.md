# Southwest Monsoon AI Platform

An end-to-end AI platform for predicting, analyzing, and querying Southwest Monsoon weather dynamics using machine learning models, Agentic RAG, and automated LaTeX report generation.

---

## 📌 Features

- **Agentic RAG Assistant:** Interactive agent powered by vector embeddings capable of querying research papers and providing cited responses.
- **Dynamic Model Evaluation:** Visualizes performance metrics and exports dynamic evaluation tables directly into LaTeX format (`metrics_table.tex`).
- **Interactive Web UI:** Built with Streamlit for seamless user interaction across models, assistant tools, and paper search.
- **Automated Workflow:** Ingestion scripts for building vector databases from academic LaTeX papers.

---

## 📁 Repository Structure

```text
Southwest-Monsoon-AI/
│
├── agents/            # AI Agent implementation and paper search tools
├── frontend/          # Streamlit user interface (app.py)
├── latex/             # Generated LaTeX tables and exported outputs
├── models/            # Trained machine learning model artifacts
├── paper/             # Research paper source files (main.tex)
├── rag/               # Retrieval-Augmented Generation ingestion scripts
├── remote_sensing/    # Remote sensing data processing modules
├── scripts/           # Utility scripts (export_latex.py)
├── tests/             # Unit and integration test suites
├── Dockerfile         # Container deployment configuration
├── README.md          # Project documentation
└── requirements.txt   # Environment dependencies