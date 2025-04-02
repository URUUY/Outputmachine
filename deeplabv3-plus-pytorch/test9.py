import os
from tqdm import tqdm
from utils.utils_metrics import compute_mIoU, show_results

if __name__ == "__main__":
    miou_mode = 2  # 只计算mIoU
    num_classes = 104  # 根据你的类别数调整
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

    # 用户提供的真实掩码和预测掩码目录
    gt_dir = "D:\FoodSeg103_img\odlabel"  # 替换为真实掩码目录路径
    pred_dir = "D:\FoodSeg103_img\m3"  # 替换为预测掩码目录路径

    # 获取所有有效样本ID（确保预测文件存在）
    image_ids = []
    for filename in os.listdir(gt_dir):
        if filename.endswith('.png'):
            image_id = os.path.splitext(filename)[0]
            pred_file = os.path.join(pred_dir, f"{image_id}_predicted3.png")
            if os.path.exists(pred_file):
                image_ids.append(image_id)
            else:
                print(f"警告: 预测文件 {pred_file} 不存在，跳过 {filename}")

    miou_out_path = "miou_out"
    os.makedirs(miou_out_path, exist_ok=True)

    if miou_mode == 0 or miou_mode == 2:
        print("正在计算mIoU...")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, 
            pred_dir, 
            image_ids, 
            num_classes, 
            name_classes,
        )
        print("mIoU计算完成。")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)