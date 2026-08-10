from ingest import collection
from config import TOP_K, SIMILARITY_THRESHOLD

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Semantic search against ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    filtered = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        similarity = 1 - dist  # convert distance to similarity
        if similarity >= SIMILARITY_THRESHOLD:
            filtered.append({
                "content": doc,
                "metadata": meta,
                "similarity": round(similarity, 4)
            })

    return filtered