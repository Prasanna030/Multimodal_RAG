import fitz
import base64
import io
from PIL import Image
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embeddings import embed_text, embed_image

def process_pdf(pdf_path):
    """Extract text & images from PDF, return documents and embeddings."""
    doc = fitz.open(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    all_docs = []
    all_embeddings = []
    image_data_store = {}

    for i, page in enumerate(doc):
        # Process text
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            for chunk in text_chunks:
                embedding = embed_text(chunk.page_content)
                all_embeddings.append(embedding)
                all_docs.append(chunk)

        # Process images
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"page_{i}_img_{img_index}"

                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64

                embedding = embed_image(pil_image)
                all_embeddings.append(embedding)

                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id}
                )
                all_docs.append(image_doc)

            except Exception as e:
                print(f"Error processing image {img_index} on page {i}: {e}")
                continue

    doc.close()
    return all_docs, all_embeddings, image_data_store
