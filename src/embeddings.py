from langchain_ollama import OllamaEmbeddings


def get_embeddings():
    """Return Ollama embeddings model."""
    return OllamaEmbeddings(model="nomic-embed-text:latest")