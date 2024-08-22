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
    
    colors = plt.cm.get_cmap('hsv', len(masks))
    
    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0]
            mask = (mask > 0.5).astype(np.uint8)
            plt.contour(mask, colors=[colors(i)])
    
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

def save_image(image, save_path):
    try:
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error saving image: {e}")

