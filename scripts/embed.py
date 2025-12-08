import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Check for GPU availability
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
    else:
        DEVICE = 'cpu'
        USE_GPU = False
        print("ℹ️  No GPU detected - using CPU")
except ImportError:
    DEVICE = 'cpu'
    USE_GPU = False
    print("ℹ️  PyTorch not found - using CPU (torch should be installed with sentence-transformers)")

PROCESSED_DIR = "data/processed"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "car_complaints"

os.makedirs(CHROMA_DIR, exist_ok=True)

def load_clean():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "complaints_clean.csv"))
    print(f"Loaded {len(df)} cleaned complaints.")
    return df


def embed_docs(docs):
    model = SentenceTransformer("BAAI/bge-large-en")
    
    # Move model to GPU if available
    if USE_GPU:
        model = model.to(DEVICE)
        print(f"🚀 Model loaded on {DEVICE.upper()}")
        # Larger batch size for GCP with T4 GPU and more RAM
        batch_size = 128
    else:
        batch_size = 32
    
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

    print("\n🎉 DONE: ChromaDB is ready!")


if __name__ == "__main__":
    main()
