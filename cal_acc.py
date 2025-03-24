import subprocess
import cv2
import numpy as np
import os
from PIL import Image
import concurrent.futures

# 输入文件夹路径
image_dir = input('Input image directory: ')  # 待处理图像的文件夹
mask_dir = input('Input mask directory: ')    # 正确掩膜的文件夹
output_dir = "outputimage"                    # 模型预测结果的保存路径
final_output_dir = "outputimg"                # 投票结果的保存路径

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_output_dir, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 定义模型预测命令
commands = [
    ['python', r'deeplabv3-plus-pytorch\predict.py', '--image'],
    ['python', r'deeplabv3-plus-pytorch copy\predict.py', '--image'],
    ['python', r'unet-pytorch\predict.py', '--image'],
    ['python', r'unet-pytorch copy\predict.py', '--image']
]

# 定义投票方式
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

# 示例 mIoU 分数
miou_scores = [18.140942278355254, 10.610629339783245, 7.081558363550649, 4.562829268445918]

# 打开结果文件
result_file = "accuracy_results2.txt"
with open(result_file, "w", encoding="utf-8") as f:
    f.write("Image\tModel1\tModel2\tModel3\tModel4\tMajorityVoting\tWeightedVoting\n")

# 处理单个图像的函数
# 修改后的 process_image 函数
def process_image(image_file):
    image_path = os.path.join(image_dir, image_file)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 调用模型进行预测
    for i, command in enumerate(commands):
        # 每次只传递一个图像路径
        current_command = command.copy()
        current_command.append(image_path)
        subprocess.run(current_command)

    # 读取预测结果
    predicted_paths = [
        os.path.join(output_dir, f"{base_name}_predicted1.jpg"),  # 模型1
        os.path.join(output_dir, f"{base_name}_predicted2.jpg"),  # 模型2
        os.path.join(output_dir, f"{base_name}_predicted3.jpg"),  # 模型3
        os.path.join(output_dir, f"{base_name}_predicted4.jpg")   # 模型4
    ]
    seg_results = [cv2.imread(path, 0) for path in predicted_paths]

    # 检查是否成功读取所有预测结果
    if any(result is None for result in seg_results):
        print(f"错误：无法读取部分分割结果图像，跳过 {image_file}。")
        return

    # 计算投票结果
    majority_result = majority_voting(seg_results)
    weighted_result = weighted_voting(seg_results, miou_scores)

    # 保存投票结果
    cv2.imwrite(os.path.join(final_output_dir, f"{base_name}_majority.jpg"), majority_result)
    cv2.imwrite(os.path.join(final_output_dir, f"{base_name}_weighted.jpg"), weighted_result)

    # 读取正确掩膜
    mask_path = os.path.join(mask_dir, f"{base_name}.png")  # 假设掩膜文件名与图像文件名相同
    if not os.path.exists(mask_path):
        print(f"错误：无法找到 {mask_path}，跳过该图像。")
        return

    true_mask = cv2.imread(mask_path, 0)
    if true_mask is None:
        print(f"错误：无法读取掩膜图像 {mask_path}，跳过该图像。")
        return

    # 计算所有结果的准确率
    accuracies = []
    for seg in seg_results:
        accuracy = np.mean(seg == true_mask) * 100
        accuracies.append(accuracy)
    majority_accuracy = np.mean(majority_result == true_mask) * 100
    weighted_accuracy = np.mean(weighted_result == true_mask) * 100

    # 将结果写入文件
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(f"{image_file}\t{accuracies[0]:.2f}\t{accuracies[1]:.2f}\t{accuracies[2]:.2f}\t{accuracies[3]:.2f}\t{majority_accuracy:.2f}\t{weighted_accuracy:.2f}\n")

    print(f"{image_file} 处理完成。")

# 使用多线程处理图像
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_image, image_files)

print("所有图像处理完成，准确率结果已保存到 accuracy_results2.txt 文件中。")