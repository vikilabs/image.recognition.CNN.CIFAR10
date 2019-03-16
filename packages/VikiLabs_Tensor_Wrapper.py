'''
    PyTorch Tensor Wrapper [ Quick Hack Functions ]

    Author:

        Vignesh Natarajan (a) Viki
        viki@vikilabs.org
'''


from __future__ import print_function
import torch
import numpy as np
import cv2
from torchvision import transforms
import VikiLabs_SimpleUI as UI
from VikiLabs_Logger import *
from PIL import Image



def TensorDetails(tensor):
    print("TENSOR > TOTAL NO ELEMENTS        : "+str(tensor.numel()))
    print("TENSOR > SHAPE                    : "+str(tensor.shape))
    print("TENSOR > DIMENSIONS | RANK | AXES : "+str(len(tensor.shape)))

'''
    Read CIFAR 32x32 png and convert to Tensor[1][3][32][32]
'''

def ReadCIFAR_ImageAsTensor(image_png):
    image = Image.open(image_png)
    t = transforms 
    tf = t.Compose([t.ToTensor(), t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    tensor_array = tf(image)
    tensor_image = tensor_array.reshape(1, 3, 32, 32)
    TensorDetails(tensor_image)
    return tensor_image
