"""
Utility functions for RAG system
"""
def hybrid_retrieve(query, vectorstore, bm25, documents, df=None, k=5):
    """
    Hybrid retrieval: combine dense vector search with BM25 keyword search
    
    Args:
        query: User query string
        vectorstore: LangChain Chroma vectorstore
        bm25: BM25Okapi instance
        documents: List of document strings
        df: Optional dataframe (for metadata, not used in current implementation)
        k: Number of results to return
    
    Returns:
        List of top k retrieved documents
    """
    # Dense vector retrieval
    dense_results = vectorstore.similarity_search_with_score(query, k=k)
    dense_docs = [doc.page_content for doc, score in dense_results]
    dense_scores = {doc.page_content: score for doc, score in dense_results}
    
    # BM25 keyword retrieval
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs = [documents[i] for i in top_bm25_indices]
    bm25_scores_dict = {documents[i]: bm25_scores[i] for i in top_bm25_indices}
    
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

