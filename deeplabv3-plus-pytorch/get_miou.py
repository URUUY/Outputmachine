import os
from PIL import Image
from tqdm import tqdm
from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

'''
Important notes about metric evaluation:
1. This script generates grayscale images. Since the values are small, they may appear nearly black when viewed as PNG - this is normal.
2. This script calculates mIoU on the validation set. The current implementation uses the test set as validation set without separate test set division.
'''

if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    #   miou_mode determines what operations to perform:
    #   0 = Full mIoU calculation (both prediction and calculation)
    #   1 = Generate predictions only
    #   2 = Calculate mIoU only (requires existing predictions)
    #---------------------------------------------------------------------------#
    miou_mode = 0
    
    #------------------------------#
    #   Number of classes + 1 (e.g., 2+1)
    #------------------------------#
    num_classes = 104
    
    #--------------------------------------------#
    #   Class names (must match json_to_dataset)
    #--------------------------------------------#
    name_classes = [
        "background", "candy", "egg tart", "french fries", "chocolate", "biscuit", 
        "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine", 
        "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", 
        "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", 
        "date", "apricot", "avocado", "banana", "strawberry", "cherry", 
        "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", 
        "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", 
        "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", 
        "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", 
        "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", 
        "noodles", "rice", "pie", "tofu", "eggplant", "potato", "garlic", 
        "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", 
        "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white radish", 
        "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", 
        "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", 
        "pepper", "green beans", "French beans", "king oyster mushroom", 
        "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom", 
        "salad", "other ingredients"
    ]
    
    #-------------------------------------------------------#
    #   Path to VOC dataset directory
    #   Default points to VOC dataset in root directory
    #-------------------------------------------------------#
    VOCdevkit_path = 'VOCdevkit'

    # Load validation image IDs
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # Generate predictions if requested
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Loading model...")
        deeplab = DeeplabV3()
        print("Model loaded successfully.")

        print("Generating predictions...")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/", image_id + ".jpg")
            image = Image.open(image_path)
            segmented_image = deeplab.get_miou_png(image)
            segmented_image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Prediction generation completed.")

    # Calculate mIoU if requested
    if miou_mode == 0 or miou_mode == 2:
        print("Calculating mIoU metrics...")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )
        print("mIoU calculation completed.")
        
        # Visualize and save results
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)