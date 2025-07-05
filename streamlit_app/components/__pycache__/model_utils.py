import torch
import clip
from PIL import Image

def load_clip_model(device='cpu'):
    """Load the CLIP model and preprocess function."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def extract_features(model, preprocess, image_path, device):
    """Extract features from an image using the CLIP model."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image).cpu()
