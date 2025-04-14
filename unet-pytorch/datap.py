import os
import cv2
from tqdm import tqdm

def resize_image(image, max_size=(650, 650)):
    """Resize image while maintaining aspect ratio and ensuring dimensions are multiples of 32.
    
    Args:
        image: Input image to resize
        max_size: Tuple (max_width, max_height) for the output size
    
    Returns:
        Resized image with dimensions as multiples of 32
    """
    h, w = image.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h, 1)  # Don't scale up if image is smaller
    new_w = int(w * scale) // 32 * 32  # Round down to nearest multiple of 32
    new_h = int(h * scale) // 32 * 32
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def process_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir):
    """Process dataset by resizing images and masks while maintaining correspondence.
    
    Args:
        image_dir: Directory containing input images
        mask_dir: Directory containing corresponding masks
        output_image_dir: Directory to save processed images (as JPG)
        output_mask_dir: Directory to save processed masks (as PNG)
    """
    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Get all PNG images in input directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    
    # Process each image with progress bar
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        
        # Skip if mask doesn't exist
        if not os.path.exists(mask_path):
            print(f"Warning: Mask for {img_name} not found, skipping...")
            continue
        
        # Read image and mask
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Skip if loading failed
        if img is None or mask is None:
            print(f"Error loading {img_name}, skipping...")
            continue
        
        # Resize both image and mask (using nearest neighbor for mask to preserve labels)
        img_resized = resize_image(img)
        mask_resized = cv2.resize(mask, (img_resized.shape[1], img_resized.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Save image as JPG with 95% quality
        new_img_name = os.path.splitext(img_name)[0] + ".jpg"
        cv2.imwrite(os.path.join(output_image_dir, new_img_name), img_resized, 
                   [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Save mask as PNG (lossless)
        cv2.imwrite(os.path.join(output_mask_dir, img_name), mask_resized)
    
    print("Processing completed.")

# Example usage
if __name__ == "__main__":
    image_folder = r"F:\Dataset\FoodSeg103_img\images\train"
    mask_folder = r"F:\Dataset\FoodSeg103_img\labels\train"
    output_image_folder = r"F:\Dataset\FoodSeg103_img\odimg"
    output_mask_folder = r"F:\Dataset\FoodSeg103_img\odlabel"

    process_dataset(image_folder, mask_folder, output_image_folder, output_mask_folder)