import streamlit as st
from src.vectorstore import get_vectorstore
from src.rag_chain import create_rag_chain
from config import DATA_DIR, CHROMA_DB_DIR

st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 RAG with Multiple PDFs")

# Initialize vectorstore
with st.spinner("Loading documents and creating embeddings..."):
    vectorstore = get_vectorstore(
        data_dir=str(DATA_DIR),
        persist_dir=str(CHROMA_DB_DIR)
    )

rag_chain = create_rag_chain(vectorstore)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("Documents")
    st.write(f"PDFs in data/: {len(list(DATA_DIR.glob('*.pdf')))}")
    if st.button("Rebuild Vector Database"):
        st.cache_resource.clear()
        st.rerun()