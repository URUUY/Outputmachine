import os
import cv2
from tqdm import tqdm

def resize_image(image, max_size=(650, 650)):
    h, w = image.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h, 1)
    new_w = int(w * scale) // 32 * 32
    new_h = int(h * scale) // 32 * 32
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def process_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask for {img_name} not found, skipping...")
            continue
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if img is None or mask is None:
            print(f"Error loading {img_name}, skipping...")
            continue
        
        img_resized = resize_image(img)
        mask_resized = cv2.resize(mask, (img_resized.shape[1], img_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        new_img_name = os.path.splitext(img_name)[0] + ".jpg"
        cv2.imwrite(os.path.join(output_image_dir, new_img_name), img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        cv2.imwrite(os.path.join(output_mask_dir, img_name), mask_resized)
    
    print("Processing completed.")

image_folder = r"F:\Dataset\FoodSeg103_img\images\train"
mask_folder = r"F:\Dataset\FoodSeg103_img\labels\train"
output_image_folder = r"F:\Dataset\FoodSeg103_img\odimg"
output_mask_folder = r"F:\Dataset\FoodSeg103_img\odlabel"

process_dataset(image_folder, mask_folder, output_image_folder, output_mask_folder)
