import numpy as np
from PIL import Image
from .config import clip_model, clip_processor
import torch

def embed_image(image_data):
    """Embed image using CLIP."""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = clip_processor(images=image, return_tensors="pt")
    features = clip_model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().numpy()

def embed_text(text):
    """Embed text using CLIP."""
    inputs = clip_processor(
        text=text, return_tensors="pt",
        padding=True, truncation=True, max_length=77
    )
    features = clip_model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().numpy()
