#----------------------------------------------------#
#   Integrates single image prediction, camera detection, and FPS testing
#   into one py file, with mode switching via the 'mode' parameter.
#----------------------------------------------------#
import time
import os
import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   To modify colors for corresponding classes, change self.colors in __init__
    #-------------------------------------------------------------------------#
    deeplab = DeeplabV3()
    #----------------------------------------------------------------------------------------------------------#
    #   mode specifies the test mode:
    #   'predict'          Single image prediction. For modifying prediction process (saving images, cropping objects etc.)
    #                      see detailed comments below
    #   'video'            Video detection, works with camera or video files. See comments below.
    #   'fps'              FPS testing, uses street.jpg from img folder. See comments below.
    #   'dir_predict'      Batch prediction on folder contents. Default scans img folder, saves to img_out.
    #   'export_onnx'      Export model to ONNX format (requires pytorch 1.7.1+)
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   count             Whether to perform pixel counting (area) and ratio calculation
    #   name_classes      Class names, same as in json_to_dataset, for printing class counts
    #
    #   count and name_classes only effective when mode='predict'
    #-------------------------------------------------------------------------#
    count           = True
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

    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["background","cat","dog"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Video file path (0 for camera)
    #                       Set to "xxx.mp4" to read xxx.mp4 from root directory
    #   video_save_path     Output video path (empty string for no save)
    #                       Set to "yyy.mp4" to save as yyy.mp4 in root directory
    #   video_fps           FPS for output video
    #
    #   video_path, video_save_path and video_fps only effective when mode='video'
    #   Need to press ctrl+c or wait until last frame for complete video save
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       Number of detections for FPS measurement (higher = more accurate)
    #   fps_image_path      Image path for FPS test
    #   
    #   test_interval and fps_image_path only effective when mode='fps'
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     Folder path for input images
    #   dir_save_path       Folder path for output images
    #   
    #   dir_origin_path and dir_save_path only effective when mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   simplify            Use Simplify onnx
    #   onnx_save_path      Output path for ONNX model
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        '''
        Notes for predict.py:
        1. This code doesn't support batch prediction directly. For batch prediction,
           use os.listdir() to traverse folder and Image.open() for prediction.
           See get_miou_prediction.py for reference implementation.
        2. Use r_image.save("img.jpg") to save results.
        3. Set blend=False to keep original and segmented images separate.
        4. To extract regions based on mask, refer to drawing section in detect_image().
           Identify pixel classes and extract corresponding regions.
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

            # Run prediction
            r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
            #r_image.show()

            # Generate save path
            save_path = os.path.join(output_dir, os.path.basename(os.path.splitext(img)[0]) + "_predicted1.png")

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
            raise ValueError("Failed to initialize camera/video. Check camera connection or video path.")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read frame
            ref, frame = capture.read()
            if not ref:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(np.uint8(frame))
            # Run detection
            frame = np.array(deeplab.detect_image(frame))
            # Convert RGB to BGR for OpenCV
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
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")