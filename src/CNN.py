
from __future__ import print_function
import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.manifold as mf
import sklearn.decomposition as dec
from VikiLabs_Logger import *
log = logger()

'''
    CIFAR Image Size = 3x32x32 [3 Depth(color), 32 width, 32 height]
    Declare A Neural Network Class
'''

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.initialize_network()

        #input_size  = 2D Array or 3D array [image]
        #output_size = 1D Array [Classified Output] 
    def initialize_network(self):
        'pixels to slide at a time 2x2'
        patch_size = 2 #2x2

        'Number of sliding operation'
        strides = 2

        '''
        conv2d(input, output, kernel)
        conv2d : input  = [ batch_size   x input_color_depth x image_height x image_width ] 
        conv2d : output = [ output_nodes x input             x height       x width       ]
        conv2d : kernel = 
        '''
        
        input  = 3
        output = 32
        kernel = 3

        self.conv1 = nn.Conv2d(input, output, kernel)
        self.pool  = nn.MaxPool2d(patch_size, strides)

        input  = 32
        output = 128
        kernel = 3
        self.conv2 = nn.Conv2d(input, output, kernel)

        input  = 128
        output = 512
        kernel = 5
        self.conv3 = nn.Conv2d(input, output, kernel)


        #Hidden Layers
        input = 512
        output = 1024

        'Fully Connected Layer 1'
        self.fc1 = nn.Linear(input, output)

        'Previous Output Dimention'
        input  = output
        output = 16

        'Fully Connected Layer 2'
        self.fc2 = nn.Linear(input, output)

        'Previous Output Dimention'
        input  = output
        output = 10

        'Fully Connected Layer 2'
        self.fc3 = nn.Linear(input, output)

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, input_tensor):
        output = self.conv1(input_tensor)
        output = F.relu(output)
        output = self.pool(output)
        output = self.dropout(output)

        output = self.conv2(output)
        output = F.relu(output)
        output = self.pool(output)
        output = self.dropout(output)

        output = self.conv3(output)
        output = F.relu(output)
        output = self.pool(output)
        output = self.dropout(output)


        #Flatten matrix excluding the batch dimention
        no_elements = self.n_elements_excluding_first_dimention(output)
        output = output.view(-1, no_elements)

        output = self.fc1(output)
        output = F.relu(output)

        output = self.fc2(output)
        output = F.relu(output)

        output = self.fc3(output)

        output = F.log_softmax(output, dim=1)

        return output

    def n_elements_excluding_first_dimention(self, tensor):
        #Shape of Tensor Skipping the First Dimention
        shape = tensor.shape[1:]  
        no_elements = 1
        
        #Number of Elements in second dimention * Number of elements in third dimension
        for e in shape:
            no_elements = no_elements * e

        return no_elements

# Call this function multiple times or manipulate n_epochs to get better  results
def train_cnn(net, training_data, batch_size, optimizer, loss_function):
    print(log._st+ "CNN TRAINING")
    net.train()  # set the net in training mode
    n_iterations = 3
    for itr in range(n_iterations):  # loop over the dataset multiple times

        current_loss = 0.0
        for i, data in enumerate(training_data, 0):
            # get inputs and labels from MNIST data
            inputs, labels = data
        
            #INPUT SIZE torch.Size([64, 3, 32, 32])

            # reset the parameter gradients
            optimizer.zero_grad()

            # FORWARD PROP + BACKWARD PROP + OPTIMIZE
            outputs = net(inputs)

            loss = loss_function(outputs, labels)

            ''' compute gradient / error'''
            loss.backward()

            ''' update weights of the network'''
            optimizer.step()

            # Compute current loss
            current_loss += loss.item()
            

            #Display results as batch of size 100
            if (i % 100) == 99:
                #start_index = i + 1
                #end_index   = i * batch_size
                current_loss = (current_loss / 100)
                #print(log._if+ 'TRAINING > START_INDEX : %d, END_INDEX : %d,  LOSS: %.3f' % (start_index, end_index, current_loss))
                print(log._if+ 'CURRENT LOSS: %.3f' % (current_loss))
                current_loss = 0.0

    print(log._ed+ "CNN TRAINING")


def evaluate_cnn(net, test_data):
    #Set Network to Evaluation Mode
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():  # Set all gradient flag to false
        for data in test_data:
            images, labels = data
            outputs = net(images)
            # return the maximum value in the array outputs.data
            # Dimention To Consider [0][1] => 1'st dimention
            # In this case it is the row of the matrix [max value in each row of the matrix]
            # _ => Max Vlue in each row of the matrix
            # predicted => array index of the max value in each row of the matrix
            dimention = 1
            _, predicted = torch.max(outputs.data, dimention)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy      = (100 * correct) / total
    images_tested = total
    print(log._if+ "[ Network Accuracy : %d ] [No Images Tested : %d ] %%" % (accuracy, images_tested))


def test_cnn(net, test_data):
    # iterate over the test set
    dataiter = iter(test_data)
    images, labels = dataiter.next()
    print(log._if+ str(images.type()))
    num_disp = 5
    # perform a prediction for a few examples
    net.eval()  # set the net in prediction mode

    with torch.no_grad():  # don't keep track of gradients
        # output contains certainties for each class
        print(log._if+ str(images.shape))
        print(log._if+ str(images[0:num_disp, :, :, :].shape))
        outputs = net(images[0:num_disp, :, :, :])

    # get the index of the highest certainty for each point
    _, predicted = torch.max(outputs.data, 1)

    fig, axes = plt.subplots(1, num_disp, figsize=(12, 8))  # , sharex=True, sharey=True)

    for i in range(0, num_disp):
        axes[i].imshow((images[i, 0, :, :] / 2 + 0.5).numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title("GT: " + str(labels[i].item()) + "; Pred: " + str(predicted[i].item()))

    plt.tight_layout()
    plt.show()


def ClassifyImage(net, image):
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #Set Net in evaluation mode [ sets the training to False ]

    net.eval()  

    with torch.no_grad():  # don't keep track of gradients
        # output contains certainties for each class
        #print(log._if+ str(image.shape))
        outputs = net(image[:, :, :, :])

    # get the index of the highest certainty for each point
    _, predicted = torch.max(outputs.data, 1)

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))  # , sharex=True, sharey=True)

    axes.imshow((image[0, 0, :, :]).numpy(), cmap='gray', vmin=0, vmax=1)
    axes.axis('off')
    axes.set_title("Prediction: " + str(predicted.item()))
    
    return classes[predicted.item()]

def Optimizer_SGD(parameters, learning_rate, momentum):
    o = optim.SGD(parameters, lr=learning_rate, momentum=momentum)
    return o

def Optimizer_ADAM(parameters, learning_rate, betas):
    o = optim.Adam(parameters, lr=learning_rate, betas=betas)
    return o

''' optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))'''

def SaveModel(model, file_name):
    torch.save(model, file_name)

def LoadModel(file_name):
    # Model class must be defined somewhere
    model = torch.load(file_name)
    model.eval()
    return model
