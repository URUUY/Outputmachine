import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


#-----------------------------------------------------------------------------------#
#   To use your trained model for prediction, you need to modify 3 parameters
#   model_path, backbone and num_classes all need to be modified!
#   If shape mismatch occurs, pay attention to modifying model_path, backbone and num_classes during training
#-----------------------------------------------------------------------------------#
class DeeplabV3(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path points to the weight file in the logs folder
        #   After training, there are multiple weight files in the logs folder, choose the one with lower validation loss.
        #   Lower validation loss doesn't necessarily mean higher miou, it only indicates better generalization performance on the validation set.
        #-------------------------------------------------------------------#
        "model_path"        : r'deeplabv3-plus-pytorch\trainedmodel\ep015-loss0.821-val_loss0.839.pth',
        #----------------------------------------#
        #   Number of classes to distinguish +1
        #----------------------------------------#
        "num_classes"       : 104,
        #----------------------------------------#
        #   Backbone network used:
        #   mobilenet
        #   xception    
        #----------------------------------------#
        "backbone"          : "mobilenet",
        #----------------------------------------#
        #   Input image size
        #----------------------------------------#
        "input_shape"       : [810, 810],
        #----------------------------------------#
        #   Downsampling factor, usually 8 or 16
        #   Should be the same as during training
        #----------------------------------------#
        "downsample_factor" : 16,
        #-------------------------------------------------#
        #   mix_type parameter controls how detection results are visualized
        #
        #   mix_type = 0 means blending original image with generated image
        #   mix_type = 1 means keeping only the generated image
        #   mix_type = 2 means removing background, keeping only targets in original image
        #-------------------------------------------------#
        "mix_type"          : 1,
        #-------------------------------#
        #   Whether to use CUDA
        #   Set to False if no GPU available
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   Initialize Deeplab
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   Set different colors for bounding boxes
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   Get model
        #---------------------------------------------------#
        self.generate()
        
        show_config(**self._defaults)
                    
    #---------------------------------------------------#
    #   Get all classifications
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #-------------------------------#
        #   Load model and weights
        #-------------------------------#
        self.net = DeepLab(num_classes=self.num_classes, backbone=self.backbone, downsample_factor=self.downsample_factor, pretrained=False)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        #   Convert image to RGB here to prevent errors when predicting grayscale images.
        #   The code only supports prediction of RGB images, all other types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   Make a copy of the input image for later drawing
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   Add gray bars to the image for non-distorting resize
        #   Or you can directly resize for recognition
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   Feed image into network for prediction
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   Get the class of each pixel
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   Crop out the gray bar parts
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   Resize the image
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   Get the class of each pixel
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        #   Counting
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            #print('-' * 63)
            #print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            #print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                #if num > 0:
                    #print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    #print('-' * 63)
                classes_nums[i] = num
            #print("classes_nums:", classes_nums)
            save_results_to_file(pr, orininal_h, orininal_w, name_classes)

        
        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   Convert new image to Image format
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   Blend new image with original image
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.uint8(pr)  # Convert pr to uint8 type
            #------------------------------------------------#
            #   Convert new image to Image format
            #------------------------------------------------#
            image = Image.fromarray(seg_img, mode='L')

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   Convert new image to Image format
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   Convert image to RGB here to prevent errors when predicting grayscale images.
        #   The code only supports prediction of RGB images, all other types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image for non-distorting resize
        #   Or you can directly resize for recognition
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   Feed image into network for prediction
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   Get the class of each pixel
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            #   Crop out the gray bar parts
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------#
                #   Feed image into network for prediction
                #---------------------------------------------------#
                pr = self.net(images)[0]
                #---------------------------------------------------#
                #   Get the class of each pixel
                #---------------------------------------------------#
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                #--------------------------------------#
                #   Crop out the gray bar parts
                #--------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
    
    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   Convert image to RGB here to prevent errors when predicting grayscale images.
        #   The code only supports prediction of RGB images, all other types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   Add gray bars to the image for non-distorting resize
        #   Or you can directly resize for recognition
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   Feed image into network for prediction
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   Get the class of each pixel
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   Crop out the gray bar parts
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   Resize the image
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   Get the class of each pixel
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
    
def save_results_to_file(pr, orininal_h, orininal_w, name_classes, output_file="result.txt"):
    num_classes = len(name_classes)
    classes_nums = np.zeros([num_classes])
    total_points_num = orininal_h * orininal_w

    with open(output_file, "w", encoding="utf-8") as f:
        header = "-" * 63 + "\n"
        title = "|{:^25} | {:^15} | {:^15}|\n".format("Key", "Value", "Ratio")
        header_line = "-" * 63 + "\n"

        f.write(header)
        f.write(title)
        f.write(header_line)

        for i in range(num_classes):
            num = np.sum(pr == i)
            ratio = num / total_points_num * 100 if total_points_num > 0 else 0

            if num > 0:
                line = "|{:^25} | {:^15} | {:^14.2f}%|\n".format(name_classes[i], num, ratio)
                separator = "-" * 63 + "\n"

                f.write(line)
                f.write(separator)

        f.write("classes_nums: " + str(classes_nums.tolist()) + "\n")

# You can call this function to save results:
# save_results_to_file(pr, orininal_h, orininal_w, name_classes)