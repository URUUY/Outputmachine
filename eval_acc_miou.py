import numpy as np
import os
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_metrics(pred_mask, true_mask, num_classes):
    """
    计算准确率和mIoU
    :param pred_mask: 预测的分割掩码 (H, W)
    :param true_mask: 真实的分割掩码 (H, W)
    :param num_classes: 类别数量
    :return: accuracy, miou
    """
    # 展平数组
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # 计算准确率
    accuracy = accuracy_score(true_flat, pred_flat)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_flat, pred_flat, labels=range(num_classes))
    
    # 计算mIoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = intersection / (union + 1e-10)  # 避免除以0
    miou = np.nanmean(iou)  # 忽略NaN值
    
    return accuracy, miou

def evaluate_models(true_mask_path, pred_mask_paths, num_classes=104):
    """
    评估多个模型
    :param true_mask_path: 真实掩码路径
    :param pred_mask_paths: 预测掩码路径列表
    :param num_classes: 类别数量
    :return: 每个模型的准确率和mIoU
    """
    # 读取真实掩码
    true_mask = cv2.imread(true_mask_path, 0)  # 以灰度模式读取
    
    if true_mask is None:
        raise ValueError(f"无法读取真实掩码文件: {true_mask_path}")
    
    results = []
    
    for i, pred_path in enumerate(pred_mask_paths, 1):
        # 读取预测掩码
        pred_mask = cv2.imread(pred_path, 0)
        
        if pred_mask is None:
            print(f"警告: 无法读取预测掩码文件: {pred_path}")
            results.append((i, 0.0, 0.0))  # 如果文件不存在，返回0
            continue
        
        # 确保预测掩码和真实掩码大小一致
        if pred_mask.shape != true_mask.shape:
            pred_mask = cv2.resize(pred_mask, (true_mask.shape[1], true_mask.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # 计算指标
        accuracy, miou = calculate_metrics(pred_mask, true_mask, num_classes)
        results.append((i, accuracy, miou))
    
    return results

# 示例使用
if __name__ == "__main__":
    # 设置路径
    image_path = input('输入测试图像路径: ')
    true_mask_path = input('输入对应的真实掩码路径: ')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 六个模型的预测结果路径
    predicted_dir = "outputimage"
    pred_mask_paths = [
        os.path.join(predicted_dir, f"{base_name}_predicted1.jpg"),  # DeepLabV3+ 模型1
        os.path.join(predicted_dir, f"{base_name}_predicted2.jpg"),  # DeepLabV3+ 模型2
        os.path.join(predicted_dir, f"{base_name}_predicted3.jpg"),  # U-Net 模型1
        os.path.join(predicted_dir, f"{base_name}_predicted4.jpg"),  # U-Net 模型2
        os.path.join(predicted_dir, f"{base_name}_predicted5.jpg"),  # 投票结果
        os.path.join(predicted_dir, f"{base_name}_predicted6.jpg")   # IR模型结果
    ]
    
    # 评估模型
    try:
        results = evaluate_models(true_mask_path, pred_mask_paths)
        
        # 打印结果
        print("\n模型评估结果:")
        print("模型编号\t准确率\t\tmIoU")
        for model_num, acc, miou in results:
            print(f"模型{model_num}\t{acc:.4f}\t\t{miou:.4f}")
        
        # 保存结果到文件
        with open("model_evaluation_results.txt", "w") as f:
            f.write("模型编号\t准确率\t\tmIoU\n")
            for model_num, acc, miou in results:
                f.write(f"模型{model_num}\t{acc:.4f}\t\t{miou:.4f}\n")
        
        print("\n评估结果已保存到 model_evaluation_results.txt")
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")