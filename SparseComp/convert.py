import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from conversion import save_compressed_model

class Baseline_compressed(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same")
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.bn6 = nn.BatchNorm2d(num_features=128)
        self.dropout3 = nn.Dropout(p=0.4)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(nn.Linear(in_features=2048, out_features=12, bias=False),
                                 nn.Linear(in_features=12, out_features=128))
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
        self.bn7 = nn.BatchNorm1d(num_features=128)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=128, out_features=10)
        
        
    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        outputs.append(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        outputs.append(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        outputs.append(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        outputs.append(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        x = self.bn5(x)
        outputs.append(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.bn6(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        outputs.append(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.bn7(x)
        x = self.dropout4(x)
        scores = self.fc5(x)
        
        return scores, outputs

model = Baseline().to(device)
print('Expected model keys: \n',model.state_dict().keys())  # Expected keys
print('loaded model keys: \n',torch.load("compressed_model_933.h5").keys())  # Loaded keys
model.load_state_dict(torch.load("/home/mal/DScale/freeml/FreeML/SparseComp/compressed_model_933.h5", map_location='cpu'))


# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
])

# Download dataset and get a single sample
dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
single_sample, _ = dataset[0]  # Extract first sample (image, label)

# Add batch dimension
single_sample = single_sample.unsqueeze(0)  # Shape: (1, 3, 32, 32)

save_compressed_model(model, 'csr', input_data=single_sample)
