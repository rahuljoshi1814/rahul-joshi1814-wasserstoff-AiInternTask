import cv2
import numpy as np
import matplotlib.pyplot as plt

def postprocess_image(image_tensor):
    image_np = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    return image_np

def visualize_segmentation(original_image, masks, scores, threshold=0.5):
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image_rgb)
    
    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0]
            mask = (mask > 0.5).astype(np.uint8)
            plt.contour(mask, colors=[np.random.rand(3,)])
    
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

def save_image(image, save_path):
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
