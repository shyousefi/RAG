from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


def load_pdfs_from_directory(data_dir: str) -> List[Document]:
    """Load all PDFs from directory and split into chunks."""
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks