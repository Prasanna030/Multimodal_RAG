import os
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
import torch
from langchain.chat_models import init_chat_model

# Load .env variables
load_dotenv()

# Set OpenAI key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Initialize LLM
llm = init_chat_model("openai:gpt-4.1")

# Ensure torch does not use gradients
torch.set_grad_enabled(False)
