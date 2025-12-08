"""
Quick test script for RAG retrieval logic with fake data
Run this while your real data is still downloading
"""
import os

import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# Create test data directory
TEST_DIR = "test_data"
CHROMA_TEST_DIR = "test_chroma_store"
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(CHROMA_TEST_DIR, exist_ok=True)

# Create fake complaint data
fake_complaints = [
    "The steering wheel makes a loud grinding noise when turning left. The dealer said it's normal but it's getting worse.",
    "Engine stalls randomly at stop signs. No warning lights. Happened three times this month.",
    "Brakes feel spongy and require more pressure to stop. Brake fluid was checked and is fine.",
    "Transmission shifts roughly between 2nd and 3rd gear. Makes a clunking sound.",
    "Air conditioning stops working after 30 minutes of driving. Works fine when first started.",
    "Check engine light came on. Code P0420. Dealer says it's the catalytic converter.",
    "Windshield wipers stop working in heavy rain. Motor seems to be failing.",
    "Car pulls to the right when braking. Alignment was checked and is correct.",
    "Radio display goes blank randomly. Sound still works but can't see station or time.",
    "Driver side window won't roll up. Motor makes noise but window doesn't move."
]

# Create fake metadata
fake_metadata = [
    {"make": "TOYOTA", "model": "CAMRY", "year": 2020},
    {"make": "HONDA", "model": "CIVIC", "year": 2019},
    {"make": "FORD", "model": "F-150", "year": 2021},
    {"make": "CHEVROLET", "model": "SILVERADO", "year": 2020},
    {"make": "TESLA", "model": "MODEL 3", "year": 2022},
    {"make": "BMW", "model": "3 SERIES", "year": 2019},
    {"make": "AUDI", "model": "A4", "year": 2021},
    {"make": "MERCEDES-BENZ", "model": "C-CLASS", "year": 2020},
    {"make": "NISSAN", "model": "ALTIMA", "year": 2019},
    {"make": "HYUNDAI", "model": "ELANTRA", "year": 2021},
]

# Save fake data to CSV (like clean.py output)
df = pd.DataFrame({
    "summary": fake_complaints,
    "make": [m["make"] for m in fake_metadata],
    "model": [m["model"] for m in fake_metadata],
    "year": [m["year"] for m in fake_metadata],
})
df.to_csv(os.path.join(TEST_DIR, "complaints_clean.csv"), index=False)
print(f"✅ Created test CSV with {len(fake_complaints)} fake complaints")

# Create embeddings and ChromaDB
print("📦 Creating embeddings (this will download BAAI/bge-large-en on first run)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    encode_kwargs={'normalize_embeddings': True}
)

# Create ChromaDB collection
client = chromadb.PersistentClient(path=CHROMA_TEST_DIR)
collection = client.get_or_create_collection(name="test_complaints")

# Embed documents
doc_embeddings = embeddings.embed_documents(fake_complaints)

# Add to ChromaDB
ids = [f"test_{i}" for i in range(len(fake_complaints))]
collection.add(
    documents=fake_complaints,
    metadatas=fake_metadata,
    embeddings=doc_embeddings,
    ids=ids
)
print(f"✅ Added {len(fake_complaints)} documents to ChromaDB")

# Create LangChain vectorstore
vectorstore = Chroma(
    client=client,
    collection_name="test_complaints",
    embedding_function=embeddings
)

# Set up BM25
tokenized_docs = [doc.lower().split() for doc in fake_complaints]
bm25 = BM25Okapi(tokenized_docs)
print("✅ Set up BM25 index")

# Import retrieval function from shared module
from utils import hybrid_retrieve

# Test queries
print("\n" + "="*60)
print("🧪 TESTING RETRIEVAL")
print("="*60)

test_queries = [
    "steering problems",
    "engine issues",
    "brake problems",
    "transmission",
    "air conditioning"
]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    retrieved = hybrid_retrieve(query, vectorstore, bm25, fake_complaints, df=None, k=3)
    for i, doc in enumerate(retrieved, 1):
        print(f"\n  Result {i}:")
        print(f"  {doc[:100]}...")

print("\n" + "="*60)
print("✅ Retrieval test complete!")
print("="*60)
print("\n💡 Next steps:")
print("   1. Test with real data once download/embedding completes")
print("   2. Test with LLM by running: streamlit run app.py")
print("   3. Clean up test data: rm -rf test_data test_chroma_store")
