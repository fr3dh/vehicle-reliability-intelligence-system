"""
Utility functions for RAG system
"""
import pandas as pd
from rank_bm25 import BM25Okapi

def hybrid_retrieve(query, vectorstore, bm25, documents, df=None, k=5, filter_dict=None):
    """
    Hybrid retrieval: combine dense vector search with BM25 keyword search
    
    Args:
        query: User query string
        vectorstore: LangChain Chroma vectorstore
        bm25: BM25Okapi instance
        documents: List of document strings
        df: Optional dataframe (for metadata filtering)
        k: Number of results to return
        filter_dict: Optional dict with filter criteria (e.g., {"year": {"$gte": 2023}, "make": "NISSAN"})
    
    Returns:
        List of top k retrieved documents
    """
    # Build ChromaDB filter from filter_dict
    chroma_filter = None
    if filter_dict:
        chroma_filter = {}
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                # Handle operators like {"$gte": 2023}
                chroma_filter[key] = value
            else:
                # Simple equality
                chroma_filter[key] = value
    
    # Dense vector retrieval with filter
    if chroma_filter:
        dense_results = vectorstore.similarity_search_with_score(query, k=k*2, filter=chroma_filter)
    else:
        dense_results = vectorstore.similarity_search_with_score(query, k=k*2)
    dense_docs = [doc.page_content for doc, score in dense_results]
    dense_scores = {doc.page_content: score for doc, score in dense_results}
    
    # Filter dataframe for BM25 if filter_dict provided
    filtered_documents = documents
    filtered_bm25 = bm25
    
    if filter_dict and df is not None:
        mask = pd.Series([True] * len(df))
        for key, value in filter_dict.items():
            if key in df.columns:
                if isinstance(value, dict):
                    # Handle operators
                    if "$gte" in value:
                        mask &= df[key] >= value["$gte"]
                    if "$lte" in value:
                        mask &= df[key] <= value["$lte"]
                    if "$gt" in value:
                        mask &= df[key] > value["$gt"]
                    if "$lt" in value:
                        mask &= df[key] < value["$lt"]
                else:
                    # Simple equality
                    mask &= df[key] == value
        
        filtered_df = df[mask].reset_index(drop=True)
        filtered_documents = filtered_df["summary"].tolist()
        
        # Rebuild BM25 on filtered documents
        tokenized_filtered = [doc.lower().split() for doc in filtered_documents]
        filtered_bm25 = BM25Okapi(tokenized_filtered)
    
    # BM25 keyword retrieval
    tokenized_query = query.lower().split()
    bm25_scores = filtered_bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [filtered_documents[i] for i in top_bm25_indices]
    bm25_scores_dict = {filtered_documents[i]: bm25_scores[i] for i in top_bm25_indices}
    
    # Combine and deduplicate
    combined_docs = {}
    for doc in dense_docs:
        combined_docs[doc] = {'dense': dense_scores.get(doc, 0), 'bm25': bm25_scores_dict.get(doc, 0)}
    for doc in bm25_docs:
        if doc not in combined_docs:
            combined_docs[doc] = {'dense': dense_scores.get(doc, 0), 'bm25': bm25_scores_dict.get(doc, 0)}
    
    # Simple fusion: normalize and combine scores
    # You can adjust weights here (0.7 dense + 0.3 bm25)
    final_scores = {}
    for doc, scores in combined_docs.items():
        # Normalize scores (simple min-max, adjust as needed)
        dense_norm = 1 / (1 + abs(scores['dense'])) if scores['dense'] != 0 else 0
        bm25_norm = scores['bm25'] / max(bm25_scores_dict.values()) if bm25_scores_dict.values() else 0
        final_scores[doc] = 0.7 * dense_norm + 0.3 * bm25_norm
    
    # Get top k combined results
    top_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [doc for doc, score in top_docs]

