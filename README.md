# Vehicle Reliability Intelligence System (Llama 3)

Streamlit + Ollama RAG app for vehicle reliability insights (dense + BM25 hybrid retrieval). Data pipeline runs outside Docker; the app can run inside Docker.

## Components
- `scripts/download.py` – fetch raw complaints
- `scripts/clean.py` – clean & normalize to CSV
- `scripts/embed.py` – full embedding (large model, for GCP/GPU)
- `scripts/embed_local.py` – smaller model, optional doc limit (for local testing)
- `app.py` – Streamlit RAG UI

## Run pipeline (host VM or local)
1) Install deps (recommend venv): `pip install -r requirements.txt`
2) Download: `python scripts/download.py`
3) Clean: `python scripts/clean.py`
4) Embed:
   - Full (GPU/GCP): `python scripts/embed.py`
   - Local test: `python scripts/embed_local.py` (set `MAX_DOCS=None` for full run)

Outputs:
- `data/processed/complaints_clean.csv`
- `chroma_store/` (ChromaDB with embeddings)

## Run the app in Docker (uses host-generated data)
1) Build: `./docker-startup build`
2) Run: `./docker-startup deploy`
   - Note: `-v $PWD:/root` is in the script so the container sees `data/` and `chroma_store/`.
3) Open: http://localhost:8501

GPU run in Docker (if you have GPU locally): `./docker-startup deploy-gpu`

## References
- [Langchain embeddings](https://python.langchain.com/v0.1/docs/integrations/text_embedding/ollama/)
- [Ollama embeddings](https://ollama.com/blog/embedding-models)
- Data source: NHTSA vehicle complaints API (model years, makes/models, complaints) via:
  - Model years: https://api.nhtsa.gov/products/vehicle/modelYears?issueType=c
  - Makes by year: https://api.nhtsa.gov/products/vehicle/makes
  - Models by make/year: https://api.nhtsa.gov/products/vehicle/models
  - Complaints: https://api.nhtsa.gov/complaints/complaintsByVehicle
