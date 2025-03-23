import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def mask_white_space(image_path, save_path, white_threshold=245):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect white areas
    mask = cv2.inRange(img, white_threshold, 255)
    
    # Morphological operations to remove small white spots (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Invert mask to keep the content and ignore white spaces
    inverted_mask = cv2.bitwise_not(mask)
    
    # Apply the mask to the original image to remove white areas
    result = cv2.bitwise_and(img, img, mask=inverted_mask)
    
    # Save the result as an image
    result_pil = Image.fromarray(result)
    result_pil.save(save_path)

# Folder paths for images
input_folder = Path("/home1/asbhide/OCT/test/NORMAL")
output_folder = Path("/home1/asbhide/OCT/test/NORMAL_crop")
output_folder.mkdir(exist_ok=True, parents=True)

# Process each image in the folder
for image_path in input_folder.glob("*.jpeg"):  # Use *.jpg or other format if needed
    save_path = output_folder / image_path.name
    mask_white_space(str(image_path), str(save_path))

print("White spaces masked successfully.")
