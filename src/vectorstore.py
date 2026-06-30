from langchain_chroma import Chroma
from .embeddings import get_embeddings
from .document_loader import load_pdfs_from_directory
import os


def get_vectorstore(data_dir: str = None, persist_dir: str = None):
    """Create or load vectorstore from PDFs."""
    embeddings = get_embeddings()
    
    if persist_dir and os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 3:
        # Load existing vectorstore
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        # Create new one
        chunks = load_pdfs_from_directory(data_dir)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    return vectorstore