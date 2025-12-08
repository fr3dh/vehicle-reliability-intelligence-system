"""
Local embedding script for MacBook testing
- Uses smaller model (bge-small-en) for faster processing
- Limits dataset to MAX_DOCS for quick testing
"""
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration for local testing
MAX_DOCS = 10000  # Limit dataset for local testing (set to None for full dataset)
USE_CPU = True  # Set to False to try GPU

# Check for GPU availability (optional, disabled by default)
DEVICE = 'cpu'
USE_GPU = False
if not USE_CPU:
    try:
        import torch
        if torch.cuda.is_available():
            DEVICE = 'cuda'
            USE_GPU = True
            print("✅ CUDA GPU detected - will use GPU acceleration")
        elif torch.backends.mps.is_available():
            DEVICE = 'mps'  # MPS (Apple Metal on macOS)
            USE_GPU = True
            print("✅ MPS GPU detected - will use GPU acceleration")
    except ImportError:
        pass

if not USE_GPU:
    print("ℹ️  Using CPU mode (optimized for 8GB RAM systems)")

PROCESSED_DIR = "data/processed"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "car_complaints"

os.makedirs(CHROMA_DIR, exist_ok=True)

def load_clean():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "complaints_clean.csv"))
    print(f"Loaded {len(df)} cleaned complaints.")
    
    # Limit dataset for local testing
    if MAX_DOCS and len(df) > MAX_DOCS:
        df = df.head(MAX_DOCS)
        print(f"📊 Limited to {MAX_DOCS} documents for local testing")
    
    return df


def embed_docs(docs):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("📦 Using smaller model (bge-small-en) for faster local processing")
    
    if USE_GPU:
        model = model.to(DEVICE)
        print(f"🚀 Model loaded on {DEVICE.upper()}")
        batch_size = 16
    else:
        batch_size = 32  # CPU batch size
    
    print(f"📦 Embedding {len(docs)} documents (batch size: {batch_size})...")
    embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=DEVICE if USE_GPU else None,
    )
    
    if USE_GPU:
        print(f"✅ Embedding complete using {DEVICE.upper()}")
    else:
        print("✅ Embedding complete using CPU")
    
    return embeddings


def build_chroma(docs, metas, embeddings):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    ids = [f"cmp_{i}" for i in range(len(docs))]

    # Chroma has a max batch size; add in chunks to avoid errors
    CHUNK = 4000
    for start in range(0, len(docs), CHUNK):
        end = start + CHUNK
        collection.add(
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end]
        )

    print("ChromaDB written to disk.")


def main():
    df = load_clean()
    docs = df["summary"].tolist()
    metas = df.drop(columns=["summary"]).to_dict("records")

    embeddings = embed_docs(docs)
    build_chroma(docs, metas, embeddings)

    print("\n🎉 DONE: ChromaDB is ready for local testing!")
    if MAX_DOCS:
        print(f"⚠️  Note: Only {MAX_DOCS} documents embedded (for full dataset, run embed.py on GCP)")


if __name__ == "__main__":
    main()

