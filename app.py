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
# Add CSS for cool VRIS logo
st.markdown("""
<style>
.vris-logo {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 3s ease infinite;
    text-align: center;
    margin-bottom: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Background colors coordinating with logo */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0e6ff 0%, #e6f0ff 100%) !important;
}

.main .block-container {
    background: linear-gradient(135deg, #f5f0ff 0%, #f0f5ff 50%, #e6f0ff 100%) !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-size: 1.3rem; font-weight: 600; margin-bottom: 10px;'>What's your name?</div>", unsafe_allow_html=True)
    your_name = st.text_input("Name", label_visibility="collapsed", placeholder="Enter your name")
    
    st.markdown("---")
    
    # Filter options
    st.markdown("### 🔍 Filters")
    try:
        df_temp = pd.read_csv(os.path.join(PROCESSED_DIR, "complaints_clean.csv"))
        
        # Year filter (single year, only 2021-2025, exclude 9999)
        years = sorted([int(y) for y in df_temp["year"].dropna().unique() if pd.notna(y) and 2021 <= int(y) <= 2025])
        year_filter = st.selectbox("Year", options=["All"] + years, index=0)
        
        # Make filter
        makes = sorted([m for m in df_temp["make"].dropna().unique() if pd.notna(m) and m != ""])
        make_filter = st.selectbox("Make", options=["All"] + makes, index=0)
        
        # Model filter (filtered by make if make is selected)
        if make_filter != "All":
            models_df = df_temp[df_temp["make"] == make_filter]
            models = sorted([m for m in models_df["model"].dropna().unique() if pd.notna(m) and m != ""])
        else:
            models = sorted([m for m in df_temp["model"].dropna().unique() if pd.notna(m) and m != ""])
        model_filter = st.selectbox("Model", options=["All"] + models, index=0)
        
        # Build filter dict
        filter_dict = {}
        if year_filter != "All":
            filter_dict["year"] = int(year_filter)
        
        if make_filter != "All":
            filter_dict["make"] = make_filter
        
        if model_filter != "All":
            filter_dict["model"] = model_filter
        
        # Store in session state (only if filters are actually set)
        st.session_state.filter_dict = filter_dict if filter_dict else None
        
    except Exception as e:
        st.warning(f"Could not load filter options: {e}")
        st.session_state.filter_dict = None
    
    st.markdown("---")
    st.markdown("Ask questions about vehicle reliability, common issues, and safety patterns.")

if your_name:
    st.title(f"Hi there, {your_name}! 🚗")
else:
    st.markdown('<div class="vris-logo">VRIS 🚗</div>', unsafe_allow_html=True)

st.caption("🚀 Powered by Llama 3 + Hybrid Retrieval | Analyzing real-world vehicle data")

# Sample questions
sample_questions = [
    "Is Nissan CVT transmission reliable overall?",
    "How does the 2022 BMW X3 feel in terms of cabin quietness and ride smoothness?",
    "What are some common issues for Honda Accord braking system?"
]

# Display sample question buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button(sample_questions[0], use_container_width=True):
        st.session_state.sample_question = sample_questions[0]
with col2:
    if st.button(sample_questions[1], use_container_width=True):
        st.session_state.sample_question = sample_questions[1]
with col3:
    if st.button(sample_questions[2], use_container_width=True):
        st.session_state.sample_question = sample_questions[2]

# Load retrieval system and LLM
try:
    vectorstore, bm25, documents, df = load_retrieval_system()
    llm = load_llm()
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I can help answer questions about car problems, common issues, and what to watch out for. What would you like to know?"}]
    
    if "filter_dict" not in st.session_state:
        st.session_state.filter_dict = None

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Check for sample question or chat input
    prompt = None
    if "sample_question" in st.session_state and st.session_state.sample_question:
        prompt = st.session_state.sample_question
        del st.session_state.sample_question  # Clear it after use
    else:
        prompt = st.chat_input()

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Searching and analyzing..."):
            # Get filter from session state
            current_filter = st.session_state.get("filter_dict", None)
            
            # Hybrid retrieval with filters
            retrieved_docs = hybrid_retrieve(prompt, vectorstore, bm25, documents, df, k=5, filter_dict=current_filter)
            
            # Format context
            context = "\n\n".join([f"Report {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
            
            # Create RAG prompt
            rag_prompt = f"""You are an automotive reliability assistant helping people make informed decisions about vehicles.

Answer the user's question by combining:
1. Your general knowledge about automotive reliability, brands, models, and common issues
2. Specific real-world complaint data provided below (which shows reported problems - keep in mind this is one-sided as it only captures complaints, not positive experiences)

Context from vehicle complaint database (these are reported problems):
{context}

Question: {prompt}

Your answer should:
- Start with a balanced perspective using your general knowledge
- Use the complaint data to highlight specific concerns or patterns when relevant
- Acknowledge that complaint data shows problems, not the full picture (many owners without issues don't file complaints)
- Be helpful and informative, not overly negative
- Provide actionable insights for buyers, owners, and mechanics

If the complaint data doesn't contain relevant information, still answer the question using your general knowledge, and mention that no specific complaints were found in the database for that topic.
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
