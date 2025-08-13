import numpy as np
from langchain_community.vectorstores import FAISS

def create_vector_store(docs, embeddings):
    """Create FAISS store from precomputed embeddings."""
    embeddings_array = np.array(embeddings)
    return FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(docs, embeddings_array)],
        embedding=None,
        metadatas=[doc.metadata for doc in docs]
    )
