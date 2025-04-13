import os
from tqdm import tqdm
from utils.utils_metrics import compute_mIoU, show_results

if __name__ == "__main__":
    miou_mode = 2  # Only calculate mIoU
    num_classes = 104  # Adjust based on your number of classes
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

    # User-provided ground truth and prediction mask directories
    gt_dir = "D:\FoodSeg103_img\odlabel"  # Replace with your ground truth mask directory path
    pred_dir = "D:\FoodSeg103_img\m3"  # Replace with your prediction mask directory path

    # Get all valid sample IDs (ensure prediction files exist)
    image_ids = []
    for filename in os.listdir(gt_dir):
        if filename.endswith('.png'):
            image_id = os.path.splitext(filename)[0]
            pred_file = os.path.join(pred_dir, f"{image_id}_predicted3.png")
            if os.path.exists(pred_file):
                image_ids.append(image_id)
            else:
                print(f"Warning: Prediction file {pred_file} not found, skipping {filename}")

    miou_out_path = "miou_out"
    os.makedirs(miou_out_path, exist_ok=True)

    if miou_mode == 0 or miou_mode == 2:
        print("Calculating mIoU...")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, 
            pred_dir, 
            image_ids, 
            num_classes, 
            name_classes,
        )
        print("mIoU calculation completed.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)