import os
import cv2
import numpy as np
from pathlib import Path

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def process_images(input_folder, output_folder, noise_type='gaussian'):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_file in input_path.glob("*.*"):
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        if noise_type == 'gaussian':
            noisy_image = add_gaussian_noise(image)
        else:
            raise ValueError("Unsupported noise type")
        
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), noisy_image)
        print(f"Processed: {img_file.name}")

# sample
input_folder = r"F:\Dataset\FoodSeg103_img\odimg" 
output_folder =  r"F:\Dataset\FoodSeg103_img\noise_odimg" 
process_images(input_folder, output_folder)
