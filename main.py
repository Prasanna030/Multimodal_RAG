from .pdf_processing import process_pdf
from .vectorstore import create_vector_store
from .pipeline import multimodal_pdf_rag_pipeline

if __name__ == "__main__":
    pdf_path = "multimodal_sample.pdf"
    docs, embeddings, image_data_store = process_pdf(pdf_path)
    vector_store = create_vector_store(docs, embeddings)

    queries = [
        "What does the chart on page 1 show about revenue trends?",
        "Summarize the main findings from the document",
        "What visual elements are present in the document?"
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 50)
        answer = multimodal_pdf_rag_pipeline(q, vector_store, image_data_store)
        print(f"Answer: {answer}")
        print("=" * 70)
