#--------------------------------------------#
#   This code section is used to analyze network architecture
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.unet import Unet

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 21
    backbone        = 'vgg'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(num_classes = num_classes, backbone = backbone).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   Multiply flops by 2 because profile doesn't count convolution 
    #   as two operations (multiplication and addition separately)
    #   Some papers count convolution as two operations (multiply by 2)
    #   Some papers only count multiplication operations (don't multiply by 2)
    #   This code follows YOLOX's approach (multiply by 2)
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))