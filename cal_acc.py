import subprocess
import cv2
import numpy as np
import os
from PIL import Image
import concurrent.futures

image_dir = input('Input image directory: ')  
mask_dir = input('Input mask directory: ')   
output_dir = "outputimage"                 
final_output_dir = "outputimg"              


os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_output_dir, exist_ok=True)


image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]


commands = [
    ['python', r'deeplabv3-plus-pytorch\predict.py', '--image'],
    ['python', r'deeplabv3-plus-pytorch copy\predict.py', '--image'],
    ['python', r'unet-pytorch\predict.py', '--image'],
    ['python', r'unet-pytorch copy\predict.py', '--image']
]


def majority_voting(seg_results):
    H, W = seg_results[0].shape
    num_classes = np.max(seg_results) + 1
    vote_map = np.zeros((H, W, num_classes), dtype=np.int32)
    for seg in seg_results:
        one_hot = np.eye(num_classes, dtype=np.int32)[seg]
        vote_map += one_hot
    final_result = np.argmax(vote_map, axis=-1)
    return final_result

def weighted_voting(seg_results, miou_scores):
    H, W = seg_results[0].shape
    num_classes = np.max(seg_results) + 1
    one_hot_maps = np.zeros((len(seg_results), H, W, num_classes))
    for i, seg in enumerate(seg_results):
        one_hot_maps[i] = np.eye(num_classes)[seg]
    miou_scores = np.array(miou_scores).reshape(-1, 1, 1, 1)
    weighted_probs = np.sum(one_hot_maps * miou_scores, axis=0) / np.sum(miou_scores)
    final_result = np.argmax(weighted_probs, axis=-1)
    return final_result


miou_scores = [18.140942278355254, 10.610629339783245, 7.081558363550649, 4.562829268445918]

result_file = "accuracy_results2.txt"
with open(result_file, "w", encoding="utf-8") as f:
    f.write("Image\tModel1\tModel2\tModel3\tModel4\tMajorityVoting\tWeightedVoting\n")


def process_image(image_file):
    image_path = os.path.join(image_dir, image_file)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

 
    for i, command in enumerate(commands):

        current_command = command.copy()
        current_command.append(image_path)
        subprocess.run(current_command)

    predicted_paths = [
        os.path.join(output_dir, f"{base_name}_predicted1.png"),
        os.path.join(output_dir, f"{base_name}_predicted2.png"), 
        os.path.join(output_dir, f"{base_name}_predicted3.png"), 
        os.path.join(output_dir, f"{base_name}_predicted4.png") 
    ]
    seg_results = [cv2.imread(path, 0) for path in predicted_paths]

    if any(result is None for result in seg_results):
        print(f"error, skip {image_file}。")
        return

    majority_result = majority_voting(seg_results)
    weighted_result = weighted_voting(seg_results, miou_scores)

    cv2.imwrite(os.path.join(final_output_dir, f"{base_name}_majority.png"), majority_result)
    cv2.imwrite(os.path.join(final_output_dir, f"{base_name}_weighted.png"), weighted_result)

    mask_path = os.path.join(mask_dir, f"{base_name}.png") 
    if not os.path.exists(mask_path):
        print(f"error {mask_path}，skip")
        return

    true_mask = cv2.imread(mask_path, 0)
    if true_mask is None:
        print(f"error {mask_path} skip")
        return

    accuracies = []
    for seg in seg_results:
        accuracy = np.mean(seg == true_mask) * 100
        accuracies.append(accuracy)
    majority_accuracy = np.mean(majority_result == true_mask) * 100
    weighted_accuracy = np.mean(weighted_result == true_mask) * 100

    with open(result_file, "a", encoding="utf-8") as f:
        f.write(f"{image_file}\t{accuracies[0]:.2f}\t{accuracies[1]:.2f}\t{accuracies[2]:.2f}\t{accuracies[3]:.2f}\t{majority_accuracy:.2f}\t{weighted_accuracy:.2f}\n")

    print(f"{image_file} ")

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(process_image, image_files)
