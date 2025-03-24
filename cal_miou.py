import numpy as np
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义计算 IoU 的函数
def calculate_iou(true_mask, pred_mask, num_classes):
    """
    计算 IoU（Intersection over Union）
    """
    iou_list = []
    for cls in range(num_classes):
        true_cls = (true_mask == cls)
        pred_cls = (pred_mask == cls)
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        if union == 0:
            iou = 0  # 如果 union 为 0，IoU 为 0
        else:
            iou = intersection / union
        iou_list.append(iou)
    return iou_list  # 返回每个类别的 IoU

# 定义计算 mIoU 的函数
def calculate_miou(true_mask_paths, pred_mask_paths, num_classes):
    """
    计算所有图像的 mIoU
    """
    miou_list = []
    for true_mask_path, pred_mask_path in zip(true_mask_paths, pred_mask_paths):
        # 读取真实掩膜和预测掩膜
        true_mask = np.array(Image.open(true_mask_path))
        pred_mask = np.array(Image.open(pred_mask_path))

        # 计算当前图像的 IoU
        iou_list = calculate_iou(true_mask, pred_mask, num_classes)
        miou = np.mean(iou_list)  # 计算 mIoU
        miou_list.append(miou)
    
    # 返回所有图像的 mIoU 列表
    return miou_list

# 定义输出统计结果的函数
def print_statistics(miou_list, model_name, file=None):
    """
    输出 mIoU 的统计结果
    """
    mean_miou = np.mean(miou_list)
    max_miou = np.max(miou_list)
    min_miou = np.min(miou_list)
    std_miou = np.std(miou_list)
    print(f"{model_name}:")
    print(f"  - 平均 mIoU: {mean_miou:.4f}")
    print(f"  - 最高 mIoU: {max_miou:.4f}")
    print(f"  - 最低 mIoU: {min_miou:.4f}")
    print(f"  - 标准差: {std_miou:.4f}")
    print()

    # 将结果写入文件
    if file:
        file.write(f"{model_name}:\n")
        file.write(f"  - 平均 mIoU: {mean_miou:.4f}\n")
        file.write(f"  - 最高 mIoU: {max_miou:.4f}\n")
        file.write(f"  - 最低 mIoU: {min_miou:.4f}\n")
        file.write(f"  - 标准差: {std_miou:.4f}\n\n")

# 定义处理单个图像的函数
def process_image(image_file, image_dir, mask_dir, output_dir, final_output_dir, num_classes):
    base_name = os.path.splitext(os.path.basename(image_file))[0]

    # 读取真实掩膜
    true_mask_path = os.path.join(mask_dir, f"{base_name}.png")  # 假设掩膜文件名与图像文件名相同
    if not os.path.exists(true_mask_path):
        print(f"错误：无法找到掩膜文件 {true_mask_path}，跳过该图像。")
        return None

    # 读取每个模型的预测结果
    model1_pred_path = os.path.join(output_dir, f"{base_name}_predicted1.jpg")
    model2_pred_path = os.path.join(output_dir, f"{base_name}_predicted2.jpg")
    model3_pred_path = os.path.join(output_dir, f"{base_name}_predicted3.jpg")
    model4_pred_path = os.path.join(output_dir, f"{base_name}_predicted4.jpg")
    majority_pred_path = os.path.join(final_output_dir, f"{base_name}_majority.jpg")
    weighted_pred_path = os.path.join(final_output_dir, f"{base_name}_weighted.jpg")

    # 检查预测结果是否存在
    if not all(os.path.exists(p) for p in [model1_pred_path, model2_pred_path, model3_pred_path, model4_pred_path, majority_pred_path, weighted_pred_path]):
        print(f"错误：部分预测结果文件缺失，跳过 {image_file}。")
        return None

    # 计算每个模型的 mIoU
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

# 主程序
if __name__ == "__main__":
    # 定义类别数量（根据你的数据集调整）
    num_classes = 104  # 假设有 131 个类别

    # 输入文件夹路径
    image_dir = input('Input image directory: ')  # 原始图像的文件夹
    mask_dir = input('Input mask directory: ')    # 真实掩膜的文件夹
    output_dir = "outputimage"                    # 模型预测结果的保存路径
    final_output_dir = "outputimg"                # 投票结果的保存路径

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 初始化存储每个模型的 mIoU 的列表
    model1_miou = []
    model2_miou = []
    model3_miou = []
    model4_miou = []
    majority_miou = []
    weighted_miou = []

    # 打开结果文件
    result_file = "miou_results.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        # 使用 ThreadPoolExecutor 并行处理图像
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(process_image, image_file, image_dir, mask_dir, output_dir, final_output_dir, num_classes)
                for image_file in image_files
            ]

            # 获取每个任务的结果
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    model1_miou.append(result["model1_miou"])
                    model2_miou.append(result["model2_miou"])
                    model3_miou.append(result["model3_miou"])
                    model4_miou.append(result["model4_miou"])
                    majority_miou.append(result["majority_miou"])
                    weighted_miou.append(result["weighted_miou"])

        # 输出每个模型的统计结果
        print_statistics(model1_miou, "Model 1", f)
        print_statistics(model2_miou, "Model 2", f)
        print_statistics(model3_miou, "Model 3", f)
        print_statistics(model4_miou, "Model 4", f)
        print_statistics(majority_miou, "Majority Voting", f)
        print_statistics(weighted_miou, "Weighted Voting", f)

    print(f"所有模型的 mIoU 统计结果已保存到 {result_file} 文件中。")