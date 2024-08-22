import cv2
from PIL import Image
from torchvision.transforms import functional as F

def preprocess_image(image_path, device, target_size=(800, 800)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = F.to_tensor(image).unsqueeze(0).to(device)
    return image

def resize_image(image, output_size=(224, 224)):
    return cv2.resize(image, output_size)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def normalize_image(image):
    return image / 255.0
