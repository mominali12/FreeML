import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import torchvision
from torchsummary import summary

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

import wandb
os.environ["WANDB_API_KEY"] = "a5512fb941c1661aa83748cacb0811fdccefd461"
run = wandb.init(project="DScale", name="slimmable-S_test", id="a3xiy7dc", resume="allow")

np.random.seed(42)
torch.manual_seed(42)
width_mult_list = [0.25, 0.5, 0.75, 1.0]

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class Baseline(nn.Module):
    def __init__(self, width_mult):
        super(Baseline, self).__init__()
        out_channels_1 = [int(64 * width_mult) for width_mult in width_mult_list]
        #3 input channels for conv1 because input is rgb image
        self.conv1 = SlimmableConv2d([3 for _ in range(len(width_mult_list))], out_channels_1, kernel_size=3, padding="valid",bias= False)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        out_channels_2 = [ int(128 * width_mult) for width_mult in width_mult_list]
        self.conv2 = SlimmableConv2d(out_channels_1, out_channels_2, kernel_size=3, padding="valid",bias= False)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        out_channels_3 = [ int(64 * width_mult) for width_mult in width_mult_list]
        self.conv3 = SlimmableConv2d(out_channels_2, out_channels_3, kernel_size=3, padding="valid",bias= False)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        #b,64,2,2
        self.flatten = nn.Flatten()
        #b,64x4
        out_channels_fc1 = [ int(256 * width_mult) for width_mult in width_mult_list]
        self.fc1 = SlimmableLinear([i*4 for i in out_channels_3], out_channels_fc1)

        out_channels_fc2 = [ int(64 * width_mult) for width_mult in width_mult_list]
        self.fc2 = SlimmableLinear(out_channels_fc1, out_channels_fc2)

        out_channels_fc3 = [10 for _ in range(len(width_mult_list))]
        self.fc3 = SlimmableLinear(out_channels_fc2, out_channels_fc3)
        
    def forward(self, x):
        exit_outputs = []
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        exit_outputs.append(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        exit_outputs.append(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        exit_outputs.append(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x, exit_outputs

dataset = CIFAR10(root='/home/mal/DScale/freeml/FreeML/EarlyExit/CIFAR-10/data', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='/home/mal/DScale/freeml/FreeML/EarlyExit/CIFAR-10/data', train=False, transform=ToTensor())

epochs = 100
batch_size=128
val_size = 5000
train_size = len(dataset) - val_size
wandb.config.update({"epochs": epochs, "batch_size": batch_size, "val_size": val_size, "train_size": train_size}, allow_val_change=True)
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4)

model = Baseline(width_mult=1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=3e-3, momentum=0.9)

# best_val_epoch, best_val_loss = 0, 1e6
# break_flag = 0
# for epoch in range(epochs):  # loop over the dataset multiple times
#     model.train()
#     t_loss = 0
#     correct = 0
#     total = 0
#     for width_mult in sorted(width_mult_list, reverse=True):
#         model.apply(
#         lambda m: setattr(m, 'width_mult', width_mult))

#         for i, data in enumerate(train_loader):
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs, _ = model(images)
#             loss = criterion(outputs, labels)
#             t_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         t_loss = t_loss / (i+1)
#         t_loss = round(t_loss, 5)
#         t_acc = round(100*(correct / total), 5)
#         model.eval()
#         v_loss = 0
#         correct = 0
#         total = 0
#         for i, data in enumerate(val_loader):
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs, _ = model(images)
#             loss = criterion(outputs, labels)
#             v_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         v_loss = v_loss/(i+1)
#         v_loss = round(v_loss, 5)
#         v_acc = round(100*(correct / total), 5)
#         # if v_loss <= best_val_loss:
#         torch.save(model.state_dict(), f"cifar10_baseline_Slimmable{width_mult}.h5")
#         #     best_val_epoch = epoch + 1
#         #     best_val_loss = v_loss
#         #     break_flag = 0
#         # else:
#         #     break_flag += 1
#         print(f'Epoch[{epoch+1}]: t_loss: {t_loss} t_acc: {t_acc} v_loss: {v_loss} v_acc: {v_acc}')
#         wandb.log({"epoch":epoch, f"train/loss/{width_mult}": t_loss, f"train/accuracy/{width_mult}": t_acc, f"validation/loss/{width_mult}": v_loss, f"validation/accuracy/{width_mult}": v_acc})
#         # if break_flag >9 :
#         #     break
# print('Finished Training')
# print('Best model saved at epoch: ', best_val_epoch)

model.load_state_dict(torch.load("/home/mal/DScale/freeml/FreeML/slimmable/cifar10_baseline_Slimmable1.0.h5"))
model.eval()
for width_mult in sorted(width_mult_list, reverse=True):
    model.apply(
    lambda m: setattr(m, 'width_mult', width_mult))
    #testing
    test_loss = 0
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_loss = test_loss/(i+1)
    test_loss = round(test_loss, 5)
    t_acc = round(100*(correct / total), 5)
    print(f"test/accuracy/{width_mult}", t_acc)
    wandb.log({f"test/accuracy/{width_mult}": t_acc})

