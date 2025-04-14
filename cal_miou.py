import numpy as np
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_iou(true_mask, pred_mask, num_classes):

    iou_list = []
    for cls in range(num_classes):
        true_cls = (true_mask == cls)
        pred_cls = (pred_mask == cls)
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        if union == 0:
            iou = 0 
        else:
            iou = intersection / union
        iou_list.append(iou)
    return iou_list 

def calculate_miou(true_mask_paths, pred_mask_paths, num_classes):
    miou_list = []
    for true_mask_path, pred_mask_path in zip(true_mask_paths, pred_mask_paths):
        true_mask = np.array(Image.open(true_mask_path))
        pred_mask = np.array(Image.open(pred_mask_path))
        iou_list = calculate_iou(true_mask, pred_mask, num_classes)
        miou = np.mean(iou_list)
        miou_list.append(miou)
    
    return miou_list

def print_statistics(miou_list, model_name, file=None):

    mean_miou = np.mean(miou_list)
    max_miou = np.max(miou_list)
    min_miou = np.min(miou_list)
    std_miou = np.std(miou_list)
    print(f"{model_name}:")
    print(f"  - avg mIoU: {mean_miou:.4f}")
    print(f"  - max mIoU: {max_miou:.4f}")
    print(f"  - min mIoU: {min_miou:.4f}")
    print(f"  - std: {std_miou:.4f}")
    print()

    if file:
        file.write(f"{model_name}:\n")
        file.write(f"  - avg mIoU: {mean_miou:.4f}\n")
        file.write(f"  - max mIoU: {max_miou:.4f}\n")
        file.write(f"  - min mIoU: {min_miou:.4f}\n")
        file.write(f"  - std: {std_miou:.4f}\n\n")

def process_image(image_file, image_dir, mask_dir, output_dir, final_output_dir, num_classes):
    base_name = os.path.splitext(os.path.basename(image_file))[0]

    true_mask_path = os.path.join(mask_dir, f"{base_name}.png") 
    if not os.path.exists(true_mask_path):
        print(f"error{true_mask_path}，skip。")
        return None

    model1_pred_path = os.path.join(output_dir, f"{base_name}_predicted1.png")
    model2_pred_path = os.path.join(output_dir, f"{base_name}_predicted2.png")
    model3_pred_path = os.path.join(output_dir, f"{base_name}_predicted3.png")
    model4_pred_path = os.path.join(output_dir, f"{base_name}_predicted4.png")
    majority_pred_path = os.path.join(final_output_dir, f"{base_name}_majority.png")
    weighted_pred_path = os.path.join(final_output_dir, f"{base_name}_weighted.png")

    if not all(os.path.exists(p) for p in [model1_pred_path, model2_pred_path, model3_pred_path, model4_pred_path, majority_pred_path, weighted_pred_path]):
        print(f"error, skip {image_file}。")
        return None

    model1_miou = calculate_miou([true_mask_path], [model1_pred_path], num_classes)[0]
    model2_miou = calculate_miou([true_mask_path], [model2_pred_path], num_classes)[0]
    model3_miou = calculate_miou([true_mask_path], [model3_pred_path], num_classes)[0]
    model4_miou = calculate_miou([true_mask_path], [model4_pred_path], num_classes)[0]
    majority_miou = calculate_miou([true_mask_path], [majority_pred_path], num_classes)[0]
    weighted_miou = calculate_miou([true_mask_path], [weighted_pred_path], num_classes)[0]

    return {
        "model1_miou": model1_miou,
        "model2_miou": model2_miou,
        "model3_miou": model3_miou,
        "model4_miou": model4_miou,
        "majority_miou": majority_miou,
        "weighted_miou": weighted_miou
    }

if __name__ == "__main__":
    num_classes = 104

    image_dir = input('Input image directory: ')  
    mask_dir = input('Input mask directory: ')   
    output_dir = "outputimage"                
    final_output_dir = "outputimg"            

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    model1_miou = []
    model2_miou = []
    model3_miou = []
    model4_miou = []
    majority_miou = []
    weighted_miou = []

    result_file = "miou_results.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(process_image, image_file, image_dir, mask_dir, output_dir, final_output_dir, num_classes)
                for image_file in image_files
            ]

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    model1_miou.append(result["model1_miou"])
                    model2_miou.append(result["model2_miou"])
                    model3_miou.append(result["model3_miou"])
                    model4_miou.append(result["model4_miou"])
                    majority_miou.append(result["majority_miou"])
                    weighted_miou.append(result["weighted_miou"])
        print_statistics(model1_miou, "Model 1", f)
        print_statistics(model2_miou, "Model 2", f)
        print_statistics(model3_miou, "Model 3", f)
        print_statistics(model4_miou, "Model 4", f)
        print_statistics(majority_miou, "Majority Voting", f)
        print_statistics(weighted_miou, "Weighted Voting", f)

    print(f"saved{result_file}")