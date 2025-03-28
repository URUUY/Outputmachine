import numpy as np
import os
from PIL import Image

class IOUMetric:
    """Class to calculate mean-iou using fast_hist method"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    
    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
    
    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

def load_image_as_array(image_path):
    """Load image as numpy array"""
    return np.array(Image.open(image_path))

def calculate_miou(pred_dir, gt_dir, num_classes):
    """
    Calculate mIoU for all images in the directories
    
    Args:
        pred_dir: directory containing prediction masks
        gt_dir: directory containing ground truth masks
        num_classes: number of classes in the segmentation task
    
    Returns:
        mean_iu: mean Intersection over Union
        iu: per-class IoU
        other_metrics: other evaluation metrics
    """
    # Initialize metric
    metric = IOUMetric(num_classes)
    
    # Get sorted list of files (assuming matching filenames)
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))
    
    # Check if number of files match
    assert len(pred_files) == len(gt_files), "Number of prediction and ground truth files don't match"
    
    # Process each image pair
    for pred_file, gt_file in zip(pred_files, gt_files):
        # Load images
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)
        
        pred_mask = load_image_as_array(pred_path)
        gt_mask = load_image_as_array(gt_path)
        
        # Add to metric (batch size=1 in this case)
        metric.add_batch([pred_mask], [gt_mask])
    
    # Calculate metrics
    acc, acc_cls, iu, mean_iu, fwavacc = metric.evaluate()
    
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Mean Class Accuracy: {acc_cls:.4f}")
    print(f"Frequency Weighted IoU: {fwavacc:.4f}")
    print(f"Mean IoU: {mean_iu:.4f}")
    print("Per-class IoU:")
    for i, val in enumerate(iu):
        print(f"  Class {i}: {val:.4f}")
    
    return mean_iu, iu, (acc, acc_cls, fwavacc)

# Example usage
if __name__ == "__main__":
    # Set your paths and number of classes
    PREDICTION_DIR = "path/to/predictions"
    GROUND_TRUTH_DIR = "path/to/ground_truths"
    NUM_CLASSES = 21  # Change this to your number of classes
    
    miou, per_class_iou, other_metrics = calculate_miou(
        PREDICTION_DIR, 
        GROUND_TRUTH_DIR, 
        NUM_CLASSES
    )