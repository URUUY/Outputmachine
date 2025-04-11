import subprocess
import cv2
import numpy as np
import os
import time
import os
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms

# Input image path
image_path = input('Input image filename: ')
base_name = os.path.splitext(os.path.basename(image_path))[0]

# Call predict.py and pass the image path
command1 = ['python', r'deeplabv3-plus-pytorch\predict.py', '--image', image_path]
command2 = ['python', r'deeplabv3-plus-pytorch copy\predict.py', '--image', image_path]
command3 = ['python', r'unet-pytorch\predict.py', '--image', image_path]
command4 = ['python', r'unet-pytorch copy\predict.py', '--image', image_path]

subprocess.run(command1)
subprocess.run(command2)
subprocess.run(command3)
subprocess.run(command4)

predicted_dir = "outputimage"
predicted_paths = [
    os.path.join(predicted_dir, f"{base_name}_predicted1.png"),  # DeepLabV3+ (no noise)
    os.path.join(predicted_dir, f"{base_name}_predicted2.png"),  # DeepLabV3+ (with noise)
    os.path.join(predicted_dir, f"{base_name}_predicted3.png"),  # U-Net (no noise)
    os.path.join(predicted_dir, f"{base_name}_predicted4.png")   # U-Net (with noise)
]


# Read images
result1 = cv2.imread(predicted_paths[0], 0)  # DeepLabV3+ (no noise)
result2 = cv2.imread(predicted_paths[1], 0)  # DeepLabV3+ (with noise)
result3 = cv2.imread(predicted_paths[2], 0)  # U-Net (no noise)
result4 = cv2.imread(predicted_paths[3], 0)  # U-Net (with noise)

if result1 is None or result2 is None or result3 is None or result4 is None:
    print("Error: Unable to read some segmentation result images, please check file paths and naming format.")
    exit(1)
seg_results = [result1, result2, result3, result4]

print(seg_results)

# Voting methods
def majority_voting(seg_results):
    """
    For each pixel, count the most frequently occurring class among the four models as the final prediction.
    """
    H, W = seg_results[0].shape
    num_classes = np.max(seg_results) + 1  # Assuming class labels start from 0
    # Initialize vote counting array (H, W, num_classes)
    vote_map = np.zeros((H, W, num_classes), dtype=np.int32)
    for seg in seg_results:
        # Convert segmentation results to one-hot encoding and accumulate in vote_map
        one_hot = np.eye(num_classes, dtype=np.int32)[seg]  # (H, W, num_classes)
        vote_map += one_hot
    # For each pixel, select the class with the most votes
    final_result = np.argmax(vote_map, axis=-1)
    return final_result

# (2) mIoU weighted voting function
def weighted_voting(seg_results, miou_scores):
    """
    Perform weighted voting based on each model's mIoU score to obtain the final pixel-level prediction.
    """
    H, W = seg_results[0].shape
    num_classes = np.max(seg_results) + 1
    # Convert segmentation results to one-hot encoding, shape (N, H, W, num_classes)
    one_hot_maps = np.zeros((len(seg_results), H, W, num_classes))
    for i, seg in enumerate(seg_results):
        one_hot_maps[i] = np.eye(num_classes)[seg]
    # Reshape mIoU scores for broadcasting (N, 1, 1, 1)
    miou_scores = np.array(miou_scores).reshape(-1, 1, 1, 1)
    # Calculate weighted probability map
    weighted_probs = np.sum(one_hot_maps * miou_scores, axis=0) / np.sum(miou_scores)
    # Take the class with the highest weighted probability as the final result
    final_result = np.argmax(weighted_probs, axis=-1)
    return final_result

# ---------------------------
# 6. Select voting method based on user input
# ---------------------------
print("Please select voting method:")
print("1 - Majority voting")
print("2 - mIoU weighted voting")
vote_type = input("Please enter your choice (1 or 2):")

# Example mIoU scores (adjust based on actual evaluation results)
miou_scores = [18.140942278355254, 10.610629339783245, 7.081558363550649, 4.562829268445918]

if vote_type == '1':
    final_segmentation = majority_voting(seg_results)
elif vote_type == '2':
    final_segmentation = weighted_voting(seg_results, miou_scores)
else:
    print("Invalid input, program exiting.")
    exit(1)

# ---------------------------
# 7. Save final voting segmentation result as grayscale image
# ---------------------------
# Save final result in outputimg folder with filename <base_name>_predicted5.jpg
final_output_dir = "outputimg"
if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)

ensemble_filename = os.path.join(final_output_dir, f"{base_name}_predicted5.png")
cv2.imwrite(ensemble_filename, final_segmentation)
print("Final voting segmentation result saved as:", ensemble_filename)

# ---------------------------
# 8. Calculate pixel percentage for each class in final segmentation map, sort from high to low and save to result.txt
# ---------------------------
# Define class name list
name_classes = [
    "background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream",
    "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew",
    "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana",
    "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig",
    "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage",
    "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg",
    "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant", "potato",
    "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce",
    "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick",
    "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans",
    "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom", "salad",
    "other ingredients"
]

import numpy as np

# Calculate total number of pixels in image
total_pixels = final_segmentation.size

# Calculate pixel count and percentage for each class
class_percentage = []
for idx, class_name in enumerate(name_classes):
    count = np.sum(final_segmentation == idx)
    percentage = count / total_pixels * 100
    class_percentage.append((class_name, percentage))

# Calculate pixel percentage for grayscale values 0-255
grayscale_percentage = []
for gray_value in range(256):
    count = np.sum(final_segmentation == gray_value)
    percentage = count / total_pixels * 100
    if percentage > 0:  # Only record grayscale values that appear
        grayscale_percentage.append((gray_value, percentage))

# Sort by percentage in descending order
class_percentage_sorted = sorted(class_percentage, key=lambda x: x[1], reverse=True)
grayscale_percentage_sorted = sorted(grayscale_percentage, key=lambda x: x[1], reverse=True)

# Write results to result.txt file
with open("result.txt", "w", encoding="utf-8") as f:
    #f.write("Class name\tPercentage(%)\n")
    for class_name, percentage in class_percentage_sorted:
        f.write(f"{class_name}\t{percentage:.2f}%\n")

print("Class percentage and grayscale value percentage results saved to result.txt file.")


######Add IR
image_path = image_path  # Please replace with actual image path
mask_path = os.path.join(final_output_dir, f"{base_name}_predicted5.png")
model_path = r"IRmodel\resnet50_foodseg103_50ep.pth"  # Please replace with actual model path
output_dir = "IRimage"

print(mask_path)
# Create timestamp subfolder
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(output_dir, timestamp)
os.makedirs(output_dir, exist_ok=True)

# Read original image and mask
original_image = np.array(Image.open(image_path).convert("RGB"))
mask = np.array(Image.open(mask_path))

# Get unique class values in mask
unique_classes = np.unique(mask)
# Iterate through each class to generate segmented images
# Read class percentages
ratio_threshold = 1.0  # Don't output classes below 1%
class_ratios = {}

with open("result.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue  # Ensure correct format to prevent parsing errors
        class_name, ratio = parts
        try:
            class_ratios[class_name] = float(ratio.strip("%"))
        except ValueError:
            continue  # Skip if parsing fails

for class_idx in unique_classes:
    if class_idx >= len(name_classes):  # Ensure index is within range
        continue

    class_name = name_classes[class_idx]  # Get class name
    if class_name not in class_ratios or class_ratios[class_name] < ratio_threshold:
        continue  # Skip if class percentage is below 1%

    # Create mask
    class_mask = (mask == class_idx)

    # Apply mask to original image
    masked_image = original_image.copy()
    masked_image[~class_mask] = 0  # Set non-class regions to black

    # Save segmented image using class name as filename
    output_path = os.path.join(output_dir, f"{class_idx}.png")  
    Image.fromarray(masked_image).save(output_path)

# Iterate and compare
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
 
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_path = r'IR'
data_train = datasets.ImageFolder(num_path, transform=transforms)
data_loader = DataLoader(data_train, batch_size=64, shuffle=True)

import torch
from torchvision import models

# Initialize model (must match training structure)
model = models.resnet50(pretrained=False)
num_classes = len(data_train.classes)  # Get number of classes from dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set to evaluation mode
model = model.to(device)  # Use GPU or CPU

print(num_classes)
from torchvision import transforms
from PIL import Image

# Define preprocessing (remove data augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(output_dir)
image_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Open result file
result_file = "result2.txt"  # Filename doesn't need `os.path.join`
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device
model = model.to(device)  # Ensure model is on GPU or CPU
model.eval()  # Set to inference mode

with open(result_file, "w", encoding="utf-8") as f:
    # Iterate through all images for prediction
    for image_file in image_files:
        # Load image
        image_path = os.path.join(output_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        #image.show()

        # Preprocess image
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)  # Ensure input data is on GPU or CPU

        # Model inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
        
        class_names = data_train.classes
        # Get predicted class name
        predicted_class = class_names[predicted_class_idx]
        filename_without_png = image_file.replace(".png", "")

        a = int(filename_without_png)
        b = int(predicted_class)
        print(a, b)
        f.write(f"{name_classes[a]}\t->\t{name_classes[b]}\n")
        # Write result to file
        #f.write(f"{image_file}\t{predicted_class}\n")
        #print(f"Processed: {image_file} -> Predicted class: {predicted_class}")

print(f"All image prediction results saved to: {result_file}")