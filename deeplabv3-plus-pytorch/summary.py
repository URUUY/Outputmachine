#--------------------------------------------#
#   This code is for analyzing network architecture
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 21
    backbone        = 'mobilenet'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=16, pretrained=False).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   Multiply flops by 2 because profile doesn't count convolution as two operations
    #   Some papers count multiply and add as two separate operations (multiply by 2)
    #   Some papers only count multiplication operations, ignoring addition (don't multiply by 2)
    #   This code chooses to multiply by 2, following YOLOX convention
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))