#----------------------------------------------------#
#   This script integrates three functions:
#   1. Single image prediction
#   2. Camera/video detection
#   3. FPS testing
#   Switch between modes by changing the 'mode' parameter.
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet_ONNX, Unet

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   To modify colors for different classes, change self.colors in __init__ function
    #-------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    #   'mode' specifies the test mode:
    #   'predict'         - Single image prediction. See detailed comments below for modifications like saving images or cropping objects.
    #   'video'           - Video detection (camera or video file). See comments below.
    #   'fps'             - FPS testing using img/street.jpg. See comments below.
    #   'dir_predict'     - Batch prediction on folder (default scans 'img' folder and saves to 'img_out'). See comments below.
    #   'export_onnx'     - Export model to ONNX format (requires pytorch 1.7.1+).
    #   'predict_onnx'    - Prediction using exported ONNX model (modify parameters around line 346 in unet.py's Unet_ONNX)
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   count             - Whether to count target pixels (area) and calculate ratios
    #   name_classes      - Class names (same as in json_to_dataset) for printing class counts
    #
    #   count and name_classes only work when mode='predict'
    #-------------------------------------------------------------------------#
    count           = False
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
    # name_classes    = ["background","cat","dog"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path        - Video file path (0 for camera)
    #   video_save_path   - Output video path (empty string for no saving)
    #   video_fps        - FPS for saved video
    #
    #   These parameters only work when mode='video'
    #   Note: Video saving completes only after ctrl+c or reaching last frame
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval     - Number of detections for FPS measurement (higher = more accurate)
    #   fps_image_path    - Image path for FPS testing
    #   
    #   These parameters only work when mode='fps'
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path   - Folder path for input images
    #   dir_save_path     - Output folder path for processed images
    #   
    #   These parameters only work when mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   simplify          - Use Simplify for ONNX
    #   onnx_save_path    - Path to save ONNX model
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode != "predict_onnx":
        unet = Unet()
    else:
        yolo = Unet_ONNX()

    if mode == "predict":
        '''
        Notes for predict.py:
        1. For batch prediction, use os.listdir() to traverse folder and Image.open() to process images.
           See get_miou_prediction.py for reference implementation.
        2. To save results: r_image.save("img.jpg")
        3. Set blend=False to keep original and segmented images separate
        4. To extract regions based on mask, refer to drawing section in detect_image function:
           seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
           for c in range(self.num_classes):
               seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
               seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
               seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        import argparse
        from PIL import Image
        import os
        # Create argument parser
        parser = argparse.ArgumentParser(description='Predict image using DeepLabV3+')
        parser.add_argument('--image', type=str, required=True, help='Path to the input image')

        # Parse arguments
        args = parser.parse_args()
        img = args.image

        try:
            # Open image
            image = Image.open(img)
        except Exception as e:
            print(f'Open Error! {e}')
        else:
            # Output directory
            output_dir = "outputimage"
            os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

            # Model prediction
            r_image = unet.detect_image(image, count=count, name_classes=name_classes)
            #r_image.show()

            # Generate save path
            save_path = os.path.join(output_dir, os.path.basename(os.path.splitext(img)[0]) + "_predicted3.png")

            # Save result
            r_image.save(save_path)
            print(f"Saved predicted image as {save_path}")

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read camera/video. Please check camera setup or video path.")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read frame
            ref, frame = capture.read()
            if not ref:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # Detection
            frame = np.array(unet.detect_image(frame))
            # Convert RGB to BGR for OpenCV display
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)
                
    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")