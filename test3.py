import subprocess
import cv2
import numpy as np
import os
from PIL import Image
import concurrent.futures
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict

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
    """计算准确率、各类别IoU和mIoU"""
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    
    # 准确率
    accuracy = accuracy_score(true_flat, pred_flat)
    
    # 混淆矩阵
    cm = confusion_matrix(true_flat, pred_flat, labels=range(num_classes))
    
    # 各类IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = np.nan_to_num(intersection / (union + 1e-10))
    miou = np.mean(iou)
    
    return accuracy * 100, iou * 100, miou * 100  # 转换为百分比

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

# 全局变量用于存储整体统计
global_stats = {
    'model1': defaultdict(list),
    'model2': defaultdict(list),
    'model3': defaultdict(list),
    'model4': defaultdict(list),
    'majority': defaultdict(list),
    'weighted': defaultdict(list)
}

def process_image(image_file):
    try:
        image_path = os.path.join(image_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        # 1. 运行模型预测
        processes = []
        for cmd in commands:
            full_cmd = cmd + [image_path]
            processes.append(subprocess.Popen(full_cmd))
        
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
        
        # 各模型指标
        for i, seg in enumerate(seg_results, 1):
            acc, iou, miou = calculate_metrics(seg, true_mask)
            results[f"model{i}_acc"] = acc
            results[f"model{i}_miou"] = miou
            results[f"model{i}_ious"] = iou  # 存储各类别IoU
            
            # 更新全局统计
            global_stats[f'model{i}']['acc'].append(acc)
            global_stats[f'model{i}']['miou'].append(miou)
            for cls in range(len(iou)):
                global_stats[f'model{i}'][f'cls_{cls}_iou'].append(iou[cls])
        
        # 多数投票指标
        acc, iou, miou = calculate_metrics(majority, true_mask)
        results["majority_acc"] = acc
        results["majority_miou"] = miou
        results["majority_ious"] = iou
        
        global_stats['majority']['acc'].append(acc)
        global_stats['majority']['miou'].append(miou)
        for cls in range(len(iou)):
            global_stats['majority'][f'cls_{cls}_iou'].append(iou[cls])
        
        # 加权投票指标
        acc, iou, miou = calculate_metrics(weighted, true_mask)
        results["weighted_acc"] = acc
        results["weighted_miou"] = miou
        results["weighted_ious"] = iou
        
        global_stats['weighted']['acc'].append(acc)
        global_stats['weighted']['miou'].append(miou)
        for cls in range(len(iou)):
            global_stats['weighted'][f'cls_{cls}_iou'].append(iou[cls])

        # 6. 保存单张图片结果
        with open(f"single_image_results/{base_name}_metrics.txt", "w") as f:
            f.write(f"=== 单张图片评估结果: {image_file} ===\n")
            for i in range(1, 5):
                f.write(f"Model {i} - Acc: {results[f'model{i}_acc']:.2f}%, mIoU: {results[f'model{i}_miou']:.2f}%\n")
                f.write("各类别IoU:\n")
                for cls, iou_val in enumerate(results[f'model{i}_ious"]):
                    f.write(f"  Class {cls}: {iou_val:.2f}%\n")
                f.write("\n")
            
            f.write(f"Majority Vote - Acc: {results['majority_acc']:.2f}%, mIoU: {results['majority_miou']:.2f}%\n")
            f.write("Weighted Vote - Acc: {results['weighted_acc']:.2f}%, mIoU: {results['weighted_miou']:.2f}%\n")

        # 7. 保存汇总结果
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

def save_global_stats():
    """保存整体统计结果"""
    with open("global_metrics.txt", "w") as f:
        f.write("=== 全局评估结果 ===\n")
        for model in global_stats:
            if not global_stats[model]['acc']:  # 跳过空数据
                continue
                
            f.write(f"\n{model.upper()}:\n")
            avg_acc = np.mean(global_stats[model]['acc'])
            avg_miou = np.mean(global_stats[model]['miou'])
            f.write(f"Average Acc: {avg_acc:.2f}%, Average mIoU: {avg_miou:.2f}%\n")
            
            # 各类别平均IoU
            f.write("各类别平均IoU:\n")
            cls_ious = []
            for key in global_stats[model]:
                if key.startswith('cls_'):
                    cls = int(key.split('_')[1])
                    avg_iou = np.mean(global_stats[model][key])
                    cls_ious.append((cls, avg_iou))
            
            # 按类别ID排序
            cls_ious.sort()
            for cls, avg_iou in cls_ious:
                f.write(f"  Class {cls}: {avg_iou:.2f}%\n")

# 初始化目录和文件
os.makedirs("single_image_results", exist_ok=True)
with open("accuracy_results2.txt", "w") as f:
    header = "Image\t"
    header += "\t".join([f"Model{i}_Acc\tModel{i}_mIoU" for i in range(1,5)])
    header += "\tMajority_Acc\tMajority_mIoU\tWeighted_Acc\tWeighted_mIoU\n"
    f.write(header)

# 并行处理
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(process_image, image_files)

# 保存全局统计
save_global_stats()
print("全部处理完成！全局结果已保存到 global_metrics.txt")