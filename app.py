import os

import chromadb
import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from rank_bm25 import BM25Okapi

from utils import hybrid_retrieve

# Configuration
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "car_complaints"
PROCESSED_DIR = "data/processed"

# Initialize components (cached to avoid reloading)
@st.cache_resource
def load_retrieval_system():
    """Load ChromaDB and set up hybrid retrieval (dense + BM25)"""
    # Load embeddings model (same as embed.py)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load ChromaDB vector store
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    
    # Convert ChromaDB collection to LangChain format
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    
    # Load documents for BM25 (keyword search)
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "complaints_clean.csv"))
    documents = df["summary"].tolist()
    
    # Tokenize documents for BM25
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    return vectorstore, bm25, documents, df

@st.cache_resource
def load_llm():
    """Load Ollama LLM"""
    return ChatOllama(base_url="http://localhost:11434", model="llama3")

# Streamlit UI
with st.sidebar:
    your_name = st.text_input("What's your name?")
    st.markdown("### Vehicle Reliability Intelligence System")
    st.markdown("**Hybrid Retrieval:** Dense vectors + BM25 keyword search")
    st.markdown("---")
    st.markdown("Ask questions about vehicle reliability, common issues, and safety patterns.")

if your_name:
    st.title(f"Hi there, {your_name}! 🚗")
else:
    st.title("Vehicle Reliability Intelligence System 🚗")

st.caption("🚀 Powered by Llama 3 + Hybrid Retrieval | Analyzing real-world vehicle reliability data")

# Load retrieval system and LLM
try:
    vectorstore, bm25, documents, df = load_retrieval_system()
    llm = load_llm()
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I can help answer questions about car problems, common issues, and what to watch out for. What would you like to know?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Analyzing vehicle reliability data and generating insights..."):
            # Hybrid retrieval
            retrieved_docs = hybrid_retrieve(prompt, vectorstore, bm25, documents, df, k=5)
            
            # Format context
            context = "\n\n".join([f"Report {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
            
            # Create RAG prompt
            rag_prompt = f"""You are an automotive reliability assistant.

You answer questions using real-world vehicle reports, failure summaries, and reliability data retrieved from the database.

Your goals:
- Identify common issues and patterns
- Summarize safety concerns
- Explain likely causes and trends
- Provide helpful insights for buyers, owners, and mechanics

Context from vehicle reliability database:
{context}

Question: {prompt}

Your answer must:
- Be concise but helpful
- Combine multiple reports into patterns when possible
- Avoid speculation beyond the provided data
- Focus on actionable insights

If the context does not contain relevant information, say:
"No specific reports were found for that question, but here is general guidance."
"""

            # Generate response using Ollama chat model
            response_msg = llm.invoke(rag_prompt)
            response = response_msg.content if hasattr(response_msg, "content") else str(response_msg)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        
        # Show retrieved documents (optional, for debugging)
        with st.expander("📄 Source Reports"):
            for i, doc in enumerate(retrieved_docs[:3], 1):
                st.markdown(f"**Report {i}:**")
                st.text(doc)
                
except FileNotFoundError as e:
    st.error(f"❌ Data not found: {e}")
    st.info("Please run the data pipeline first:\n1. `python scripts/download.py`\n2. `python scripts/clean.py`\n3. `python scripts/embed.py`")
except Exception as e:
    st.error(f"❌ Error loading RAG system: {e}")
    st.info("Make sure:\n- ChromaDB is set up (run embed.py)\n- Ollama server is running\n- All dependencies are installed")
