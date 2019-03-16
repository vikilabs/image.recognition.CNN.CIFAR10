'''
    PyTorch Wrapper to work with CIFAR Handwritten Digit Database

    Author:

        Vignesh Natarajan (a) Viki
        viki@vikilabs.org
'''


import torchvision
from torchvision import transforms
import torch
from torchvision import datasets
import os
import errno
from VikiLabs_Logger import *
import matplotlib.pyplot as plt
import numpy as np

log = logger()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#classes = ('car', 'plane', 'dog', 'cat',
#           'deer', 'bird', 'frog', 'horse', 'ship', 'truck')


def Download_CIFAR_TrainingData(path):
    print(log._st+ "DOWNLOADING CIFAR TRAINING DATA")
    t = transforms 
    '''
        Convert Image from range [0, 1] to  range [-1 to 1]
        
        image = (image - n_mean)/n_std
    '''
    #Mean of all 3 channels (depth, height, width)
    #n_mean = (0.5, 0.5, 0.5)
    #n_std  = (0.5, 0.5, 0.5)

    tf = t.Compose([t.ToTensor(), t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #tf = t.Compose([t.ToTensor(), t.Normalize(n_mean, n_std)])
    data_object = datasets.CIFAR10(path, train=True, download=True, transform=tf)
    print(log._ed+ "DOWNLOADING CIFAR TRAINING DATA")
    return data_object


def Download_CIFAR_TestData(path):
    print(log._st+ "DOWNLOADING CIFAR TEST DATA")
    t = transforms 
    
    '''
        Convert Image from range [0, 1] to  range [-1 to 1]
        
        image = (image - n_mean)/n_std
    '''
    n_mean = (0.5, 0.5, 0.5)
    n_std  = (0.5, 0.5, 0.5)

    tf = t.Compose([t.ToTensor(), t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #tf = t.Compose([t.ToTensor(), t.Normalize(n_mean, n_std)])
    data_object  = datasets.CIFAR10(path, train=False, download=True, transform=tf)
    print(log._ed+ "DOWNLOADING CIFAR TEST DATA")
    return data_object

def Load_CIFAR_Data(data_object, batch_size):
    print(log._st+ "LOADING CIFAR DATA")
    tud = torch.utils.data
    data = tud.DataLoader(data_object, batch_size=batch_size, shuffle=True, num_workers=2)
    print(log._ed+ "LOADING CIFAR DATA")
    return data 

def Show_CIFAR_SAMPLE_Images(training_data):
    dataiter = iter(training_data)
    images, labels = dataiter.next()

    num_sample_images = 5
    fig, axes = plt.subplots(1, num_sample_images, figsize=(8, 6))
    
    for i in range(0, num_sample_images):
        axes[i].imshow(np.transpose((images[i,:,:,:]/2 + 0.5).numpy(), (1, 2, 0)), vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(classes[labels[i].item()])
        print(labels[i].item())


    plt.tight_layout()
    plt.show()

def save_image(numpy_array, file_name):
    image_name = file_name + str(".png")
    tensor_array = torch.from_numpy(numpy_array)
    torchvision.utils.save_image(tensor_array, image_name)

def StoreDataAsImage(cifar_data, dfolder):
    
    try:
        os.mkdir(dfolder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    file_base = "number"
    
    '''
    CIFAR training data has 938 records. Each record in CIFAR has the following
        1. images of shape [64, 3, 32, 32]  -> 64 handwritten digits
        2. labels for images of shape [64]  -> 64 label for the 64 handwritten digit images
    '''
    '''Full Download : ??'''
    #no_records_to_store = len(cifar_data)

    '''Only 64 Images Download'''
    no_records_to_store = 1

    #Iterate Over CIFAR DATA
    for i, data in enumerate(cifar_data, 0):
        
        if(i >= no_records_to_store):
            break

        images, labels = data

        for j in range(len(images)):
            file_name = dfolder+str("/")+file_base+"_"+str(labels[j].item())+"_"+str(i)+"_"+str(j)              

            '''
            Pixel Values will be in range between -1 and 1
            '''
            n_std = 0.5
            n_mean = 0.5
            normalized_image   = images[i,:,:,:]
            denormalized_image = ((normalized_image * n_std) + n_mean).numpy()
            image_np_array = np.transpose(denormalized_image, (1, 2, 0))
            '''
            Pixel Values will be in range between 0 and 1
            '''
            save_image(image_np_array, file_name)

'''
cifar_data_path = './data'
image_path = './images'
training_batch_size = 64
test_batch_size = 1000

training_object = Download_CIFAR_TrainingData(cifar_data_path)
test_object     = Download_CIFAR_TestData(cifar_data_path)

training_data   = Load_CIFAR_Data( training_object, training_batch_size   )
test_data       = Load_CIFAR_Data( test_object,     test_batch_size       )

Show_CIFAR_SAMPLE_Images(training_data)

#StoreDataAsImage(training_data, image_path)
'''
