import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import wandb
os.environ["WANDB_API_KEY"] = "a5512fb941c1661aa83748cacb0811fdccefd461"
run = wandb.init(project="DScale", name="EarlyExit-S")

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding="valid", bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="valid", bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="valid", bias=False)
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
            pooled_outs.append(pool_3d(out))
        x = torch.cat(pooled_outs, dim=1)
        x = self.flatten(x)
        x = self.dropout(x)
        scores = self.fc(x)
        return scores
    
def simulate_exit(x, choice=None):
    if choice is not None:
        choice = choice
    else:
        choice = np.random.choice(np.arange(0, 3), p=[0.34, 0.33, 0.33])
    batch_size = x[0].shape[0]
    reshaped_output = []
    for i in range(3):
        if i <= choice :
            reshaped_output.append(x[i])
        else:
            reshaped_output.append(torch.zeros_like(x[i]))
    return reshaped_output, choice

def train(baseline, exit_model, layer, epochs, criterion, 
          optimizer, train_loader, val_loader, model_name, gen_ee=False):
    best_val_epoch, best_val_loss = 0, 1e6
    break_flag = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        exit_model.train()
        t_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outs = baseline(images)
            if gen_ee:
                outs, choice = simulate_exit(outs)
                out = outs
            else:
                out = outs[layer]
            outputs = exit_model(out)
            loss = criterion(outputs, labels)
            t_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        t_loss = t_loss / (i+1)
        t_loss = round(t_loss, 5)
        t_acc = round(100*(correct / total), 5)
        exit_model.eval()
        v_loss = 0
        correct = 0
        total = 0
        choice=0
        for i, data in enumerate(val_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            _, outs = baseline(images)
            if gen_ee:
                outs, choice = simulate_exit(outs)
                out = outs
            else:
                out = outs[layer]
            outputs = exit_model(out)
            loss = criterion(outputs, labels)
            v_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        v_loss = v_loss/(i+1)
        v_loss = round(v_loss, 5)
        v_acc = round(100*(correct / total), 5)
        # if v_loss <= best_val_loss:
        #     torch.save(exit_model.state_dict(), model_name)
        #     best_val_epoch = epoch + 1
        #     best_val_loss = v_loss
        #     break_flag = 0
        # else:
        #     break_flag += 1
        print(f'Epoch[{epoch+1}]: t_loss: {t_loss} t_acc: {t_acc} v_loss: {v_loss} v_acc: {v_acc}')
        wandb.log({"epoch":epoch, f"train/loss/{choice}": t_loss, f"train/accuracy/{choice}": t_acc, f"validation/loss/{choice}": v_loss, f"validation/accuracy/{choice}": v_acc})
        # if break_flag >19 :
        #     break
    print('Finished Training')
    print('Best model saved at epoch: ', best_val_epoch)

dataset = CIFAR10(root='/home/mal/DScale/freeml/FreeML/EarlyExit/CIFAR-10/data', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='/home/mal/DScale/freeml/FreeML/EarlyExit/CIFAR-10/data', train=False, transform=ToTensor())

batch_size=128
val_size = 5000
pct = .01
train_size = int(pct*len(dataset))
val_size = len(dataset) - train_size
# train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
_, val_ds = random_split(val_ds, [int(0.7*len(val_ds)), len(val_ds) - int(0.7*len(val_ds))])
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4)

baseline = Baseline().to(device)
baseline.load_state_dict(torch.load("/home/mal/DScale/freeml/FreeML/slimmable/cifar10_baseline_s_new.h5", map_location='cpu'))
baseline.eval()

result_dict = {}
result_dict["accuracy"] = []
in_features = [0]
for ii, in_feature in enumerate(in_features):
    precisions, recall, f1 = [], [], []
    learning_rate = 5e-3
    exit_model = GeneralEEModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(exit_model.parameters(), lr=learning_rate, momentum=0.9)
    epochs = 100
    ee_model_name = "gen_ee_cifar10_s_new.h5"
    train(baseline, exit_model, ii, epochs, criterion, optimizer, 
          train_loader, val_loader, ee_model_name, gen_ee=True)
    exit_model.load_state_dict(torch.load(ee_model_name, map_location='cpu'))
    exit_model.eval()
    correct = 0
    total = 0#         x = self.fc1(x)
    with torch.no_grad(): 
        true_y, pred_y = [], []
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            _, outs = baseline(images)
            outs, _ = simulate_exit(outs)
            outputs = exit_model(outs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_y = pred_y + list(predicted.detach().cpu().numpy())
            true_y = true_y + list(labels.detach().cpu().numpy())
        report = classification_report(true_y, pred_y, output_dict=True)
        result_dict["accuracy"].append(report["accuracy"])
        for i, (key, value) in enumerate(report.items()):
            if i < 10:
                precisions.append(value['precision'])
                recall.append(value['recall'])
                f1.append(value['f1-score'])
        result_dict[ii] = {
            'precision': precisions,
            'recall': recall,
            'f1': f1,
            'conf_mat': confusion_matrix(true_y, pred_y)
        }




criterion = nn.CrossEntropyLoss()
exit_model = GeneralEEModel().to(device)
exit_model.load_state_dict(torch.load("/home/mal/DScale/freeml/FreeML/slimmable/gen_ee_cifar10_s_new.h5", map_location='cpu'))
exit_model.eval()
v_loss = 0
correct = 0
total = 0
choice=0
# for one
for i, data in enumerate(test_loader):
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    _, outs = baseline(images)
    if True:
        outs, choice = simulate_exit(outs, choice)
        out = outs
    outputs = exit_model(out)
    loss = criterion(outputs, labels)
    v_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
v_loss = v_loss/(i+1)
v_loss = round(v_loss, 5)
v_acc = round(100*(correct / total), 5)
print("test/accuracy_exitlayer_0", v_acc)
wandb.log({"test/accuracy_exitlayer_0": v_acc})


v_loss = 0
correct = 0
total = 0
choice=1
for i, data in enumerate(test_loader):
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    _, outs = baseline(images)
    if True:
        outs, choice = simulate_exit(outs, choice)
        out = outs
    outputs = exit_model(out)
    loss = criterion(outputs, labels)
    v_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
v_loss = v_loss/(i+1)
v_loss = round(v_loss, 5)
v_acc = round(100*(correct / total), 5)
print("test/accuracy_exitlayer_1", v_acc)
wandb.log({"test/accuracy_exitlayer_1": v_acc})

v_loss = 0
correct = 0
total = 0
choice=2
for i, data in enumerate(test_loader):
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    _, outs = baseline(images)
    if True:
        outs, choice = simulate_exit(outs, choice)
        out = outs
    outputs = exit_model(out)
    loss = criterion(outputs, labels)
    v_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
v_loss = v_loss/(i+1)
v_loss = round(v_loss, 5)
v_acc = round(100*(correct / total), 5)
print("test/accuracy_exitlayer_2", v_acc)
wandb.log({"test/accuracy_exitlayer_2": v_acc})
