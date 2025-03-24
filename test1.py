import subprocess
import cv2
import numpy as np
import os

# 输入图片地址
image_path = input('Input image filename: ')
base_name = os.path.splitext(os.path.basename(image_path))[0]

# 调用 predict.py 并传递图片地址
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
    os.path.join(predicted_dir, f"{base_name}_predicted1.jpg"),  # DeepLabV3+（无噪声）
    os.path.join(predicted_dir, f"{base_name}_predicted2.jpg"),  # DeepLabV3+（有噪声）
    os.path.join(predicted_dir, f"{base_name}_predicted3.jpg"),  # U-Net（无噪声）
    os.path.join(predicted_dir, f"{base_name}_predicted4.jpg")   # U-Net（有噪声）
]


# 读取图像
result1 = cv2.imread(predicted_paths[0], 0)  # DeepLabV3+（无噪声）
result2 = cv2.imread(predicted_paths[1], 0)  # DeepLabV3+（有噪声）
result3 = cv2.imread(predicted_paths[2], 0)  # U-Net（无噪声）
result4 = cv2.imread(predicted_paths[3], 0)  # U-Net（有噪声）

if result1 is None or result2 is None or result3 is None or result4 is None:
    print("错误：无法读取部分分割结果图像，请检查文件路径和文件名格式。")
    exit(1)
seg_results = [result1, result2, result3, result4]

print(seg_results)

#以下是投票方式
def majority_voting(seg_results):
    """
    对每个像素，统计四个模型中出现次数最多的类别作为最终预测结果。
    """
    H, W = seg_results[0].shape
    num_classes = np.max(seg_results) + 1  # 假设类别标签从 0 开始
    # 初始化投票计数数组 (H, W, num_classes)
    vote_map = np.zeros((H, W, num_classes), dtype=np.int32)
    for seg in seg_results:
        # 将分割结果转换为 one-hot 编码，并累加到 vote_map 中
        one_hot = np.eye(num_classes, dtype=np.int32)[seg]  # (H, W, num_classes)
        vote_map += one_hot
    # 对每个像素选择投票数最多的类别
    final_result = np.argmax(vote_map, axis=-1)
    return final_result

# (2) mIoU 加权投票函数
def weighted_voting(seg_results, miou_scores):
    """
    根据每个模型的 mIoU 分数进行加权投票，得到最终像素级预测结果。
    """
    H, W = seg_results[0].shape
    num_classes = np.max(seg_results) + 1
    # 将分割结果转换为 one-hot 编码，形状为 (N, H, W, num_classes)
    one_hot_maps = np.zeros((len(seg_results), H, W, num_classes))
    for i, seg in enumerate(seg_results):
        one_hot_maps[i] = np.eye(num_classes)[seg]
    # 调整 mIoU 分数形状以便广播 (N, 1, 1, 1)
    miou_scores = np.array(miou_scores).reshape(-1, 1, 1, 1)
    # 计算加权概率图
    weighted_probs = np.sum(one_hot_maps * miou_scores, axis=0) / np.sum(miou_scores)
    # 取加权后概率最大的类别作为最终结果
    final_result = np.argmax(weighted_probs, axis=-1)
    return final_result

# ---------------------------
# 6. 根据用户输入选择投票方式
# ---------------------------
print("请选择投票方式：")
print("1 - 多数投票")
print("2 - mIoU 加权投票")
vote_type = input("请输入选择（1 或 2）：")

# 示例 mIoU 分数（根据实际评估结果调整）
miou_scores = [13.967509525840141, 5.41968874237658, 4.099860584546196, 2.700358527367753]

if vote_type == '1':
    final_segmentation = majority_voting(seg_results)
elif vote_type == '2':
    final_segmentation = weighted_voting(seg_results, miou_scores)
else:
    print("无效输入，程序退出。")
    exit(1)

# ---------------------------
# 7. 保存最终的投票分割结果为灰度图
# ---------------------------
# 将最终结果保存在 outputimg 文件夹下，文件名为 <base_name>_predicted5.jpg
final_output_dir = "outputimg"
if not os.path.exists(final_output_dir):
    os.makedirs(final_output_dir)

ensemble_filename = os.path.join(final_output_dir, f"{base_name}_predicted5.jpg")
cv2.imwrite(ensemble_filename, final_segmentation)
print("最终投票分割结果已保存为：", ensemble_filename)

# ---------------------------
# 8. 计算每个类别在最终分割图中的像素占比，并按从多到少排序保存到 result.txt 文件
# ---------------------------
# 定义类别名称列表
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

# 计算图像总像素数
total_pixels = final_segmentation.size

# 计算每个类别的像素数量及占比
class_percentage = []
for idx, class_name in enumerate(name_classes):
    count = np.sum(final_segmentation == idx)
    percentage = count / total_pixels * 100
    class_percentage.append((class_name, percentage))

# 计算灰度值 0-255 的像素占比
grayscale_percentage = []
for gray_value in range(256):
    count = np.sum(final_segmentation == gray_value)
    percentage = count / total_pixels * 100
    if percentage > 0:  # 只记录出现的灰度值
        grayscale_percentage.append((gray_value, percentage))

# 按照占比从大到小排序
class_percentage_sorted = sorted(class_percentage, key=lambda x: x[1], reverse=True)
grayscale_percentage_sorted = sorted(grayscale_percentage, key=lambda x: x[1], reverse=True)

# 将结果写入 result.txt 文件
with open("result.txt", "w", encoding="utf-8") as f:
    f.write("类别名称\t占比(%)\n")
    for class_name, percentage in class_percentage_sorted:
        f.write(f"{class_name}\t{percentage:.2f}%\n")

    f.write("\n灰度值\t占比(%)\n")
    for gray_value, percentage in grayscale_percentage_sorted:
        f.write(f"{gray_value}\t{percentage:.2f}%\n")

print("类别占比和灰度值占比结果已保存到 result.txt 文件中。")


