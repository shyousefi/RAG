from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(vectorstore, model_name: str = "llama3.2:1b-instruct-fp16"):
    """Create RAG chain."""
    llm = ChatOllama(model=model_name, temperature=0.7)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    system_prompt = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If you cannot find the answer in the context, say you don't know. "
        "Be clear and concise.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain