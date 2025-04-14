import os
from PIL import Image
from tqdm import tqdm
from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

'''
Important Notes for Metric Evaluation:
1. This script generates grayscale images where values are small. When viewed as JPG, 
   they may appear nearly black - this is normal.
2. The script calculates mIoU on the validation set. In this implementation, 
   the test set is used as the validation set (no separate test set is defined).
3. Only models trained on VOC-format data can use this script for mIoU calculation.
'''

if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    # miou_mode controls what operations to perform:
    # 0 = Full pipeline (get predictions + calculate mIoU)
    # 1 = Only get predictions
    # 2 = Only calculate mIoU
    #---------------------------------------------------------------------------#
    miou_mode = 0
    
    #------------------------------#
    # Number of classes + background
    # Example: 2 classes + 1 background = 3
    #------------------------------#
    num_classes = 21
    
    #--------------------------------------------#
    # Class names - must match json_to_dataset
    #--------------------------------------------#
    name_classes = [
        "background", "aeroplane", "bicycle", "bird", "boat", 
        "bottle", "bus", "car", "cat", "chair", "cow", 
        "diningtable", "dog", "horse", "motorbike", "person", 
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    # Alternative example:
    # name_classes = ["_background_", "cat", "dog"]
    
    #-------------------------------------------------------#
    # Path to VOC dataset directory
    # Default points to VOCdevkit in root directory
    #-------------------------------------------------------#
    VOCdevkit_path = 'VOCdevkit'

    # Load validation image IDs
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    # Prediction generation mode
    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Loading model...")
        unet = Unet()
        print("Model loaded successfully.")

        print("Generating predictions...")
        for image_id in tqdm(image_ids, desc="Processing images"):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            try:
                image = Image.open(image_path)
                prediction = unet.get_miou_png(image)
                prediction.save(os.path.join(pred_dir, image_id + ".png"))
            except Exception as e:
                print(f"Error processing {image_id}: {str(e)}")
                continue
        print("Prediction generation completed.")

    # mIoU calculation mode
    if miou_mode == 0 or miou_mode == 2:
        print("Calculating mIoU metrics...")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, 
            pred_dir, 
            image_ids, 
            num_classes, 
            name_classes
        )
        print("mIoU calculation completed.")
        
        # Visualize and save results
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)