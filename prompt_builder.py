from langchain.schema.messages import HumanMessage

def create_multimodal_message(query, retrieved_docs, image_data_store):
    """Prepare GPT-4V input with both text and images."""
    content = [{"type": "text", "text": f"Question: {query}\n\nContext:\n"}]

    text_docs = [d for d in retrieved_docs if d.metadata.get("type") == "text"]
    image_docs = [d for d in retrieved_docs if d.metadata.get("type") == "image"]

    if text_docs:
        text_context = "\n\n".join([
            f"[Page {d.metadata['page']}]: {d.page_content}" for d in text_docs
        ])
        content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})

    for d in image_docs:
        img_id = d.metadata.get("image_id")
        if img_id in image_data_store:
            content.append({"type": "text", "text": f"\n[Image from page {d.metadata['page']}]:\n"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[img_id]}"}
            })

    content.append({"type": "text", "text": "\n\nPlease answer based on text and images."})
    return HumanMessage(content=content)
