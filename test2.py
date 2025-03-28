import subprocess
import cv2
import numpy as np
import os
from PIL import Image
import concurrent.futures
from sklearn.metrics import accuracy_score, confusion_matrix

# 输入文件夹路径
image_dir = input('Input image directory: ')
mask_dir = input('Input mask directory: ')
output_dir = "outputimage"
final_output_dir = "outputimg"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_output_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

commands = [
    ['python', r'deeplabv3-plus-pytorch\predict.py', '--image'],
    ['python', r'deeplabv3-plus-pytorch copy\predict.py', '--image'],
    ['python', r'unet-pytorch\predict.py', '--image'],
    ['python', r'unet-pytorch copy\predict.py', '--image']
]

def calculate_metrics(pred, true, num_classes=104):
    """计算准确率和mIoU"""
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    # 准确率
    accuracy = accuracy_score(true_flat, pred_flat)
    
    # mIoU
    cm = confusion_matrix(true_flat, pred_flat, labels=range(num_classes))
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = np.nan_to_num(intersection / (union + 1e-10))
    miou = np.mean(iou)
    
    return accuracy * 100, miou * 100  # 转换为百分比

def majority_voting(seg_results):
    H, W = seg_results[0].shape
    num_classes = np.max(seg_results) + 1
    vote_map = np.zeros((H, W, num_classes), dtype=np.int32)
    for seg in seg_results:
        one_hot = np.eye(num_classes, dtype=np.int32)[seg]
        vote_map += one_hot
    return np.argmax(vote_map, axis=-1)

def weighted_voting(seg_results, weights):
    H, W = seg_results[0].shape
    num_classes = np.max(seg_results) + 1
    one_hot_maps = np.zeros((len(seg_results), H, W, num_classes))
    for i, seg in enumerate(seg_results):
        one_hot_maps[i] = np.eye(num_classes)[seg]
    weights = np.array(weights).reshape(-1, 1, 1, 1)
    weighted_probs = np.sum(one_hot_maps * weights, axis=0) / np.sum(weights)
    return np.argmax(weighted_probs, axis=-1)

miou_weights = [18.14, 10.61, 7.08, 4.56]  # 示例权重

def process_image(image_file):
    try:
        image_path = os.path.join(image_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        # 1. 运行模型预测
        processes = []
        for cmd in commands:
            full_cmd = cmd + [image_path]
            processes.append(subprocess.Popen(full_cmd))
        
        # 等待所有预测完成
        for p in processes:
            p.wait()
            if p.returncode != 0:
                print(f"警告: {image_file} 的模型预测失败")
                return

        # 2. 读取预测结果
        pred_paths = [
            os.path.join(output_dir, f"{base_name}_predicted1.png"),
            os.path.join(output_dir, f"{base_name}_predicted2.png"),
            os.path.join(output_dir, f"{base_name}_predicted3.png"),
            os.path.join(output_dir, f"{base_name}_predicted4.png")
        ]
        
        seg_results = []
        for path in pred_paths:
            if not os.path.exists(path):
                print(f"警告: 预测结果 {path} 不存在")
                return
            pred = cv2.imread(path, 0)
            if pred is None:
                print(f"警告: 无法读取预测结果 {path}")
                return
            seg_results.append(pred)

        # 3. 读取真实掩码
        mask_path = os.path.join(mask_dir, f"{base_name}.png")
        if not os.path.exists(mask_path):
            print(f"警告: 真实掩码 {mask_path} 不存在")
            return
            
        true_mask = cv2.imread(mask_path, 0)
        if true_mask is None:
            print(f"警告: 无法读取真实掩码 {mask_path}")
            return

        # 调整大小确保一致
        for i in range(len(seg_results)):
            if seg_results[i].shape != true_mask.shape:
                seg_results[i] = cv2.resize(seg_results[i], 
                                          (true_mask.shape[1], true_mask.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

        # 4. 计算投票结果
        majority = majority_voting(seg_results)
        weighted = weighted_voting(seg_results, miou_weights)
        
        cv2.imwrite(os.path.join(final_output_dir, f"{base_name}_majority.png"), majority)
        cv2.imwrite(os.path.join(final_output_dir, f"{base_name}_weighted.png"), weighted)

        # 5. 计算指标
        results = {}
        for i, seg in enumerate(seg_results, 1):
            acc, miou = calculate_metrics(seg, true_mask)
            results[f"model{i}_acc"] = acc
            results[f"model{i}_miou"] = miou
        
        acc, miou = calculate_metrics(majority, true_mask)
        results["majority_acc"] = acc
        results["majority_miou"] = miou
        
        acc, miou = calculate_metrics(weighted, true_mask)
        results["weighted_acc"] = acc
        results["weighted_miou"] = miou

        # 6. 保存结果
        with open("accuracy_results2.txt", "a") as f:
            line = f"{image_file}\t"
            line += "\t".join([f"{results[f'model{i}_acc']:.2f}" for i in range(1,5)])
            line += "\t" + "\t".join([f"{results[f'model{i}_miou']:.2f}" for i in range(1,5)])
            line += f"\t{results['majority_acc']:.2f}\t{results['majority_miou']:.2f}"
            line += f"\t{results['weighted_acc']:.2f}\t{results['weighted_miou']:.2f}\n"
            f.write(line)

        print(f"处理完成: {image_file}")
        
    except Exception as e:
        print(f"处理 {image_file} 时出错: {str(e)}")

# 初始化结果文件
with open("accuracy_results2.txt", "w") as f:
    header = "Image\t"
    header += "\t".join([f"Model{i}_Acc\tModel{i}_mIoU" for i in range(1,5)])
    header += "\tMajority_Acc\tMajority_mIoU\tWeighted_Acc\tWeighted_mIoU\n"
    f.write(header)

# 并行处理
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(process_image, image_files)

print("全部处理完成！")