# RAG System with Ollama + ChromaDB
A simple, modular Retrieval-Augmented Generation (RAG) system built with Python, Streamlit, LangChain, and local Ollama models.

## Features
- Load multiple PDFs from `data/` folder
- Automatic text splitting and embedding
- Persistent vector database with ChromaDB
- Clean Streamlit UI with chat history
- Fully modular code structure
- Local LLM (no API keys needed)

## Tech Stack
- **LLM**: Ollama (llama3.2:1b)
- **Embedding**: Ollama (nomic-embed-text)
- **Vector Store**: ChromaDB
- **Framework**: LangChain + Streamlit
- **Documents**: PyPDF

## Setup
```bash
# 1. Clone the repo
# 2. Create environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull required Ollama models
ollama pull nomic-embed-text:latest
ollama pull llama3.2:1b-instruct-fp16

# 5. Put your English PDF files inside the data/ folder

# 6. Run the app
streamlit run app.py
