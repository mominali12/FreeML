import torch.nn as nn
import torch.nn.functional as F
from sparsecomp import compress_NN_models
import torch
import os
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Baseline(nn.Module):
#     def __init__(self):
#         super(Baseline, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding="same")
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.bn1 = nn.BatchNorm2d(num_features=32)
#         self.bn2 = nn.BatchNorm2d(num_features=32)
#         self.dropout1 = nn.Dropout(p=0.2)
        
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.bn3 = nn.BatchNorm2d(num_features=64)
#         self.bn4 = nn.BatchNorm2d(num_features=64)
#         self.dropout2 = nn.Dropout(p=0.3)
        
#         self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same")
#         self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same")
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
#         self.bn5 = nn.BatchNorm2d(num_features=128)
#         self.bn6 = nn.BatchNorm2d(num_features=128)
#         self.dropout3 = nn.Dropout(p=0.4)
        
#         self.flatten = nn.Flatten()
        
#         self.fc1 = nn.Linear(in_features=2048, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=128)
#         self.fc3 = nn.Linear(in_features=128, out_features=128)
#         self.fc4 = nn.Linear(in_features=128, out_features=128)
#         self.bn7 = nn.BatchNorm1d(num_features=128)
#         self.dropout4 = nn.Dropout(p=0.5)
#         self.fc5 = nn.Linear(in_features=128, out_features=10)
        
        
#     def forward(self, x):
#         outputs = []
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.bn1(x)
#         outputs.append(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.bn2(x)
#         x = self.pool1(x)
#         x = self.dropout1(x)
#         outputs.append(x)
        
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.bn3(x)
#         outputs.append(x)
#         x = self.conv4(x)
#         x = F.relu(x)
#         x = self.bn4(x)
#         x = self.pool2(x)
#         x = self.dropout2(x)
#         outputs.append(x)
        
#         x = self.conv5(x)
#         x = F.relu(x)
#         x = self.bn5(x)
#         outputs.append(x)
#         x = self.conv6(x)
#         x = F.relu(x)
#         x = self.bn6(x)
#         x = self.pool3(x)
#         x = self.dropout3(x)
#         outputs.append(x)
        
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.relu(x)
#         x = self.fc4(x)
#         x = F.relu(x)
#         x = self.bn7(x)
#         x = self.dropout4(x)
#         scores = self.fc5(x)
        
#         return scores, outputs

#Baseline S
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding="valid")
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="valid")
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="valid")
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        
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

# for M_earlyexit_model
# class GeneralEEModel(nn.Module):
#     def __init__(self):
#         super(GeneralEEModel, self).__init__()
#         self.avg_ = nn.AvgPool2d(kernel_size=2)
#         self.upsample_1 = nn.Upsample(scale_factor=2)
#         self.upsample_2 = nn.Upsample(scale_factor=4)
        
#         self.conv1x1 = nn.Conv2d(in_channels=320, out_channels=16, kernel_size=1, padding="same")
#         self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding="same")
#         self.pool = nn.MaxPool2d(kernel_size=2)
# #         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding="same")
        
#         self.bn1 = nn.BatchNorm2d(num_features=16)
#         self.bn2 = nn.BatchNorm2d(num_features=64)
# #         self.bn3 = nn.BatchNorm2d(num_features=64)
        
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(in_features=1024, out_features=10)
    
#     def forward(self, x):
#         resized_outputs = []
#         resized_outputs.append(self.avg_(x[0]))
#         resized_outputs.append(x[1])
#         resized_outputs.append(x[2])
#         resized_outputs.append(self.upsample_1(x[3]))
#         resized_outputs.append(self.upsample_1(x[4]))
# #         resized_outputs.append(self.upsample_2(x[5]))
#         x = torch.cat(resized_outputs, dim=1)
#         x = self.conv1x1(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = self.bn1(x)
# #         x = self.conv1(x)
# #         x = F.relu(x)
# #         x = self.bn2(x)
# #         x = self.conv2(x)
# #         x = F.relu(x)
# #         x = self.bn3(x)
#         x = self.flatten(x)
#         scores = self.fc1(x)
#         return scores


#for S_earlyexit_model
class GeneralEEModel(nn.Module):
    def __init__(self):
        super(GeneralEEModel, self).__init__()
        self.pool_kernels = [
            (1, 6, 6), (1, 3, 3), (1, 1, 1)
        ]
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=10)
    
    def forward(self, x, inference=False):
        pooled_outs = []
        for layer, out in enumerate(x):
            pool_3d = nn.MaxPool3d(kernel_size=self.pool_kernels[layer])
            pooled_outs.append(pool_3d(x))
        x = torch.cat(pooled_outs, dim=1)
        x = self.flatten(x)
        x = self.dropout(x)
        scores = self.fc(x)
        return scores

# Define your model (assuming GeneralEEModel is defined elsewhere)
model = Baseline().to(device)
# model = GeneralEEModel().to(device)


# Load CIFAR-10 dataset
dataset = CIFAR10(root='./data', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor())

# Define data loaders
batch_size = 128
val_size = 5000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size * 2, num_workers=4)

# Define other parameters
target_size = 200  # Target size in KB
num_epochs = 10
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
regularizerParam = 0.0
compressionStep = 0.1
initialCompressionStep = 0.1
fastCompression = False
modelName = "S_earlyexit_compressed_model"
device = "cpu"
accuracyAware = True
layersFactorization = True
calculateInputs = None

# Call the compress_NN_models function
compress_NN_models(
    model, target_size, train_loader, test_loader,
    val_loader=val_loader, num_epochs=num_epochs, learning_rate=learning_rate,
    criterion=criterion, regularizerParam=regularizerParam, compressionStep=compressionStep,
    initialCompressionStep=initialCompressionStep, fastCompression=fastCompression,
    modelName=modelName, device=device, accuracyAware=accuracyAware,
    layersFactorization=layersFactorization, calculateInputs=calculateInputs
)

