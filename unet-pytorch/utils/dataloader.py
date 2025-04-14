import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    """Custom Dataset loader for U-Net segmentation model with VOC dataset"""
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train  # Flag for training vs validation mode
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        #-------------------------------#
        #   Load image and label from files
        #-------------------------------#
        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        
        #-------------------------------#
        #   Apply data augmentation if in training mode
        #-------------------------------#
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        # Preprocess image and label
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])  # HWC to CHW
        png = np.array(png)
        
        # Handle out-of-range class values (e.g., borders/edges)
        png[png >= self.num_classes] = self.num_classes
        
        #-------------------------------------------------------#
        #   Convert to one-hot encoding
        #   We use num_classes+1 because VOC dataset has border pixels
        #   that need to be ignored during training (+1 creates ignore class)
        #-------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]  # One-hot encoding
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        """Generate random float in range [a, b)"""
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        """Apply random data augmentation including:
        - Resizing with random scaling and aspect ratio
        - Random horizontal flipping
        - Random color jitter in HSV space
        - Padding to maintain input dimensions
        """
        image = cvtColor(image)  # Ensure RGB format
        label = Image.fromarray(np.array(label))  # Convert to PIL Image
        
        #------------------------------#
        #   Get original and target dimensions
        #------------------------------#
        iw, ih = image.size  # Original dimensions
        h, w = input_shape   # Target dimensions

        if not random:
            # Validation mode - simple resizing with padding
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            # Resize and pad image
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))  # Gray padding
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            # Resize and pad label
            label = label.resize((nw, nh), Image.NEAREST)  # No interpolation for labels
            new_label = Image.new('L', [w, h], (0))  # Background padding
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   Training mode - random transformations
        #------------------------------------------#
        
        # Random aspect ratio and scaling
        new_ar = iw/ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.25, 2)  # Random scale factor
        
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
            
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        
        # Random horizontal flip
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random positioning with padding
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # Color jitter in HSV space
        image_data = np.array(image, np.uint8)
        
        # Random HSV adjustments
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        
        # Convert to HSV and apply transformations
        hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(hsv)
        
        # Create lookup tables for each channel
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(image_data.dtype)  # Hue is circular in [0,180]
        lut_sat = np.clip(x * r[1], 0, 255).astype(image_data.dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(image_data.dtype)

        # Apply transformations and convert back to RGB
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), 
                              cv2.LUT(sat, lut_sat), 
                              cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


def unet_dataset_collate(batch):
    """Custom collate function for DataLoader to properly format batch data"""
    images = []
    pngs = []
    seg_labels = []
    
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
        
    # Convert lists to properly typed tensors
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()  # Class indices should be long integers
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    
    return images, pngs, seg_labels