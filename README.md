# RAG System with Ollama + ChromaDB

A simple, modular Retrieval-Augmented Generation (RAG) system built with Python, Streamlit, LangChain, and local Ollama models.

## How RAG Works (Overview)

This project follows the standard RAG pipeline:

1. **Document Loading & Chunking**  
   All PDFs in the `data/` folder are loaded and split into smaller chunks using `RecursiveCharacterTextSplitter` (chunk size = 800 characters with 100 characters overlap).

2. **Embedding**  
   Each chunk is converted into a numerical vector using the `nomic-embed-text` model via Ollama.

3. **Vector Storage**  
   The chunks and their embeddings are stored persistently in **ChromaDB**.

4. **Retrieval**  
   When a user asks a question:
   - The question is converted to an embedding.
   - The system finds the most similar chunks using **Cosine Similarity**.
   - Top 4 relevant chunks are retrieved.

5. **Generation**  
   The retrieved chunks (context) + user question are sent to the LLM (`llama3.2:1b`).
   The model generates an answer based only on the provided context.

**Key Concepts:**
- **Chunking**: Breaks long documents into manageable pieces while preserving context with overlap.
- **Retrieval**: Uses cosine similarity to find semantically similar text.
- **Augmented Generation**: LLM answers using retrieved documents instead of just its internal knowledge.

---

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
git clone https://github.com/shyousefi/RAG.git

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
