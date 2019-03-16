'''

CIFAR Image Recognization with Convolutional Neural Network

'''
import sys
sys.path.append('../packages')
import VikiLabs_SimpleUI as UI
from VikiLabs_Logger import *
from VikiLabs_CIFAR_Wrapper import *
from CNN import *
from VikiLabs_Tensor_Wrapper import *


log             = logger()
net             = None

cifar_path      = '../data'
image_path      = '../images'
model_file      = '../model/torch.pt'

training_batch_size = 64
#training_batch_size = 128
test_batch_size     = 1000
learning_rate       = 0.0001
momentum            = 0.9


def ui_callback(file_name):
    global net
    tensor = ReadCIFAR_ImageAsTensor(file_name)
    prediction   = ClassifyImage(net, tensor)
    return prediction


training_object = Download_CIFAR_TrainingData(cifar_path)
test_object     = Download_CIFAR_TestData(cifar_path)

training_data   = Load_CIFAR_Data( training_object, training_batch_size   )
test_data       = Load_CIFAR_Data( test_object,     test_batch_size       )


''' Initialize CNN '''
try:
    net = LoadModel(model_file)
except:
    pass

if not net:
    print(log._er+ "LoadModel CNN Model")
    net = CNN()
    ''' Common loss function for classification
    '''
    loss_function = nn.CrossEntropyLoss()
    parameters = net.parameters()
    betas = (0.9, 0.999)

    ''' Create an Optimizer '''
    #optimizer  = Optimizer_SGD(parameters, learning_rate, momentum)
    ''' ADAM Optimizer Learning is fast '''
    optimizer = Optimizer_ADAM(parameters, learning_rate, betas)

    train_cnn(net, training_data, training_batch_size, optimizer, loss_function)
    SaveModel(net, model_file)
else:
    print(log._if+ "LoadModel CNN Model")


UI.render(ui_callback)
