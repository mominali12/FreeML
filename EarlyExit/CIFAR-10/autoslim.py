import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import torchvision
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)

def forward_loss(model,  criterion, input, target, device=None, return_acc=False):
    """Forward pass and loss computation."""
    model.eval()
    with torch.no_grad():
        output, _ = model(input.to(device) if device else input)
        loss = criterion(output, target.to(device) if device else target)
    if return_acc:
        # Assuming classification task, computing accuracy
        acc = (output.argmax(dim=1) == target).float().mean()
        return loss, acc.item()
    return loss

def print_full_model(model):
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    kb = 1000
    model_size = 0
    params = 0
    for name, param in model.named_parameters():
        layer_size = param.nelement() * param.element_size()
        model_size += layer_size
        # print(name,"\t", param.nelement(), "\t", param.element_size(),"\t", layer_size / kb, "KB")

    for name, buffer in model.named_buffers():
        layer_size = buffer.nelement() * buffer.element_size()
        model_size += layer_size
        # print(name,"\t", layer_size / kb, "KB")
    # print("Model Size:", model_size / kb, "KB")

    params = sum(p.numel() for p in model.parameters())
    # print("Model Params:", params)

    return (model_size / kb), params

class AutoSlim_Linear(nn.Linear):
    # By Kainat
    def __init__(self, in_features, out_features, bias=True):
        super(AutoSlim_Linear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.divisor = int(self.out_features_max/10)  #make 10 groups for features pruning/slimming

    def forward(self, input, in_features=None, out_features=None):
        if in_features is None:
            in_features = self.in_features_max
        if out_features is None:
            out_features = self.out_features_max
        
        weight = self.weight[:out_features, :in_features]
        if self.bias is not None:
            bias = self.bias[:out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)

class AutoSlim_Conv2d(nn.Conv2d):
    # By Kainat
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True):
        super(AutoSlim_Conv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        # self.divisor = 8
        self.divisor = int(self.out_channels_max/10)  #make 10 groups for channels pruning/slimming

    def forward(self, input, in_channels=None , out_channels=None):
        if in_channels is None:
            in_channels = self.in_channels_max
        if  out_channels is None:
            out_channels = self.out_channels_max
            
        self.groups = in_channels if self.depthwise else 1
        weight = self.weight[:out_channels, :in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:out_channels]
        else:
            bias = self.bias
        
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)

        return y

class AutoSlimmableBaseline(nn.Module ):
    def __init__(self, channels = [[3,64],[64,128],[128,64],[64,256],[256,64],[64,10]]):
        super(AutoSlimmableBaseline, self).__init__()
        
        self.channels = channels
        
        self.conv1 = AutoSlim_Conv2d(self.channels[0][0], self.channels[0][1], kernel_size=3, padding="valid",bias= False)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = AutoSlim_Conv2d(self.channels[1][0], self.channels[1][1], kernel_size=3, padding="valid",bias= False)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = AutoSlim_Conv2d(self.channels[2][0], self.channels[2][1], kernel_size=3, padding="valid",bias= False)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        #b,64,2,2
        self.flatten = nn.Flatten()
        #b,64x4
        self.fc1 = AutoSlim_Linear(4*self.channels[3][0], self.channels[3][1])
        self.fc2 = AutoSlim_Linear(self.channels[4][0], self.channels[4][1])
        self.fc3 = AutoSlim_Linear(self.channels[5][0], self.channels[5][1])
        
    def forward(self, x):
        exit_outputs = []
        # [[3,58],[58,128],[128,64],[64,256],[256,64],[64,10]]
        
        x = self.conv1(x,self.channels[0][0], self.channels[0][1])
        x = F.relu(x)
        x = self.pool1(x)
        exit_outputs.append(x)
        
        x = self.conv2(x,self.channels[1][0], self.channels[1][1])
        x = F.relu(x)
        x = self.pool2(x)
        exit_outputs.append(x)
        
        x = self.conv3(x,self.channels[2][0], self.channels[2][1])
        x = F.relu(x)
        x = self.pool3(x)
        exit_outputs.append(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x, 4*self.channels[3][0], self.channels[3][1])
        x = self.fc2(x, self.channels[4][0], self.channels[4][1])
        x = self.fc3(x, self.channels[5][0], self.channels[5][1])
        
        return x, exit_outputs

def get_conv_layers(m):
    layers = []
    # The layer must have a width_mult attribute. The second element of the us attribute must be True(which means the layer should be scalable/ only last layer is not scalable).
    if (isinstance(m, torch.nn.Conv2d)):
        layers.append(m)
    for child in m.children():
        layers += get_conv_layers(child)
    return layers

def get_all_layers(m):
    layers = []
    # The layer must have a width_mult attribute. The second element of the us attribute must be True(which means the layer should be scalable/ only last layer is not scalable).
    if (isinstance(m, torch.nn.Conv2d)) or isinstance(m, torch.nn.Linear):
        layers.append(m)
    for child in m.children():
        layers += get_all_layers(child)
    return layers

def prune_model_post_training(loader, model, criterion,autoslim_target_sizes):
    """network pruning"""
    model.eval()
    # bn_calibration_init(model)
    # layers = get_conv_layers(model)
    layers = get_all_layers(model)
    print('Totally {} layers to slim.'.format(len(layers)))
    
    error = np.zeros(len(layers)-1)
    input, target = next(iter(loader))
    # input = input.cuda()
    # target = target.cuda()
    # model=model.cuda()

    autoslim_target_sizes = sorted(autoslim_target_sizes)
    autoslim_target_size = autoslim_target_sizes.pop()
    print('Find autoslim net at model size {}kB'.format(autoslim_target_size))
    channels = [[3,64],[64,128],[128,58],[58,256],[256,64],[64,10]]
    while True:
        model_size, params = print_full_model(AutoSlimmableBaseline(channels=channels))
        if model_size < autoslim_target_size:
            print("Model for given model_size:", model_size, "kB has channels:",channels)
            if len(autoslim_target_sizes) == 0:
                break
            else:
                autoslim_target_size = autoslim_target_sizes.pop()
                print('Find autoslim net at size {}'.format(
                    autoslim_target_size))
        else:
            # need to prune the model
            for i in range(len(layers)-1):
                torch.cuda.empty_cache()
                error[i] = 0.
                if isinstance(layers[i], AutoSlim_Linear):
                    outc = layers[i].out_features - layers[i].divisor
                    if outc <= 0 or outc > layers[i].out_features_max:
                        error[i] += 1.
                        continue
                    
                    layers[i].out_features -= layers[i].divisor
                    layers[i+1].in_features = layers[i].out_features
                    
                    channels[i][1] = outc
                    channels[i+1][0] = outc
                    
                    model.apply(lambda m: setattr(m, 'channels', channels))
                    loss, error_batch = forward_loss(
                        model, criterion, input, target, None, return_acc=True)
                    error[i] += error_batch
                    
                    layers[i].out_features += layers[i].divisor
                    layers[i+1].in_features = layers[i].out_features
                    
                    channels[i][1] = layers[i].out_features
                    channels[i+1][0] = channels[i][1]
                    model.apply(lambda m: setattr(m, 'channels', channels))
                elif isinstance(layers[i], AutoSlim_Conv2d):
                    outc = layers[i].out_channels - layers[i].divisor
                    if outc <= 0 or outc > layers[i].out_channels_max:
                        error[i] += 1.
                        continue
                    
                    layers[i].out_channels -= layers[i].divisor
                    layers[i+1].in_channels = layers[i].out_channels
                    
                    channels[i][1] = outc
                    channels[i+1][0] = outc
                    
                    model.apply(lambda m: setattr(m, 'channels', channels))
                    loss, error_batch = forward_loss(
                        model, criterion, input, target, None, return_acc=True)
                    error[i] += error_batch
                    
                    layers[i].out_channels += layers[i].divisor
                    layers[i+1].in_channels = layers[i].out_channels
                    
                    channels[i][1] = layers[i].out_channels
                    channels[i+1][0] = channels[i][1]
                    model.apply(lambda m: setattr(m, 'channels', channels))
                
            best_index = np.argmin(error)
            
            if isinstance(layers[best_index], AutoSlim_Linear):
                layers[best_index].out_features -= layers[best_index].divisor
                layers[best_index+1].in_features = layers[best_index].out_features
                channels[best_index][1] = layers[best_index].out_features
                channels[best_index+1][0] = channels[best_index][1]
                print(
                    'Adjust layer {} for {} to {}, error: {}.'.format(
                        best_index, -layers[best_index].divisor,
                        layers[best_index].out_features, error[best_index]))
                
            elif isinstance(layers[best_index], AutoSlim_Conv2d):
                layers[best_index].out_channels -= layers[best_index].divisor
                layers[best_index+1].in_channels = layers[best_index].out_channels
                channels[best_index][1] = layers[best_index].out_channels
                channels[best_index+1][0] = channels[best_index][1]

                print(
                    'Adjust layer {} for {} to {}, error: {}.'.format(
                        best_index, -layers[best_index].divisor,
                        layers[best_index].out_channels, error[best_index]))
    return

dataset = CIFAR10(root='/data22/mal/data/CIFAR10', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='/data22/mal/data/CIFAR10', train=False, transform=ToTensor())

batch_size=128
val_size = 5000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4)


# model = AutoSlimmableBaseline(channels=[[3, 32], [32, 72], [72, 40], [40, 256], [256, 32], [32, 10]])
# model = AutoSlimmableBaseline(channels=[[3, 16], [16, 72], [72, 40], [40, 256], [256, 32], [32, 10]])
# model = AutoSlimmableBaseline(channels=[[3, 8], [8, 72], [72, 40], [40, 256], [256, 32], [32, 10]])
model = AutoSlimmableBaseline()
criterion = nn.CrossEntropyLoss()
# model.load_state_dict(torch.load("/data22/kal/code/FreeML/EarlyExit/CIFAR-10/models/cifar10_auto_slim_s_20E_last.h5"))

# 30555540
prune_model_post_training(val_loader, model, criterion, [512, 256, 128, 64, 32])