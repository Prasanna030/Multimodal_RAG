from .vectorstore import create_vector_store
from .prompt_builder import create_multimodal_message
from .embeddings import embed_text
from .config import llm

def retrieve_multimodal(query, vector_store, k=5):
    """Retrieve documents for a query."""
    query_embedding = embed_text(query)
    return vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)

def multimodal_pdf_rag_pipeline(query, vector_store, image_data_store):
    """Run retrieval and query GPT-4V."""
    context_docs = retrieve_multimodal(query, vector_store)
    message = create_multimodal_message(query, context_docs, image_data_store)
    response = llm.invoke([message])

    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  - Text from page {page}: {preview}")
        else:
            print(f"  - Image from page {page}")
    return response.content
