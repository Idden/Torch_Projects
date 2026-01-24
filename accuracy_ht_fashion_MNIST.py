import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

class HadamardConv2D(nn.Module):

    def __init__(self, dim=4, stride=1, out_channels=16, coeff_idx=None):
        super().__init__()

        assert (dim * dim & (dim * dim - 1)) == 0  # make sure dim^2 is power of 2

        self.dim = dim
        self.stride = stride
        self.out_channels = out_channels

        n = dim * dim
        n_qubits = int(np.log2(n))

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        H = np.asarray(Operator(qc).data, dtype=np.float32)
        self.register_buffer("H", torch.from_numpy(H))

        if coeff_idx is None:
            coeff_idx = torch.arange(out_channels)
        else:
            coeff_idx = torch.tensor(coeff_idx, dtype=torch.long)
            
        self.register_buffer("coeff_idx", coeff_idx)

    def forward(self, x):

        B, C, height, width = x.shape
        assert C == 1  # assume 1 input channel into HT

        patches = F.unfold(x, kernel_size=self.dim, stride=self.stride)
        patches = patches.transpose(1, 2)
        #n = self.dim * self.dim

        y = patches @ self.H.t()

        y = y[:, :, self.coeff_idx]

        # reshape back to feature map
        out_h = (height - self.dim) // self.stride + 1
        out_w = (width - self.dim) // self.stride + 1
        y = y.transpose(1, 2).reshape(B, self.out_channels, out_h, out_w)

        return y
    
# download format
# turns MNIST images to PyTorch tensors and normalizes between [-1,1] centered at 0
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=1),
    #transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# download data to computer
raw_train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

train_eval_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=test_transform
)

raw_test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,            # testing data so labels are unknown during training
    download=True,
    transform=test_transform
)

# use raw_train_small and raw_test_small to decrease number of training and test images
# use raw_train_dataset and raw_test_dataset for entire dataset
#raw_train_subset = Subset(raw_train_dataset, range(10000))
#raw_test_subset = Subset(raw_test_dataset, range(1000))

train_dataset = raw_train_dataset
test_dataset  = raw_test_dataset

# loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=256,         # each epoch is 128 samples
    shuffle=True,            # randomize after each training epoch
    num_workers=4,
    pin_memory=True
)

train_eval_loader = DataLoader(
    train_eval_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1024,         # each epoch is 256 samples
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# test shapes of pytorch datasets
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

# CNN model
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.hconv = HadamardConv2D(dim=4, stride=1, out_channels=16)
        self.mix = nn.Conv2d(16, 32, kernel_size=1, bias=False)

        self.conv1a = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv4a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4b = nn.Conv2d(256, 256, 3, padding=1)

        # a and b batch norms for their corresponding conv layer because batchnorm learns
        self.bn0 = nn.BatchNorm2d(32)

        self.bn1a = nn.BatchNorm2d(32)
        self.bn2a = nn.BatchNorm2d(64)
        self.bn3a = nn.BatchNorm2d(128)
        self.bn4a = nn.BatchNorm2d(256)

        self.bn1b = nn.BatchNorm2d(32)
        self.bn2b = nn.BatchNorm2d(64)
        self.bn3b = nn.BatchNorm2d(128)
        self.bn4b = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.drop = nn.Dropout(0.3)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        #self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.hconv(x)
        x = self.mix(x)
        x = F.relu(self.bn0(x))

        x = F.relu(self.bn1a(self.conv1a(x)))
        x = self.pool(F.relu(self.bn1b(self.conv1b(x))))

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.pool(F.relu(self.bn2b(self.conv2b(x))))

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.pool(F.relu(self.bn3b(self.conv3b(x))))

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))

        #x = torch.flatten(x, start_dim=1)

        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        #x = self.drop(x)
        #x = self.fc3(x)

        return x

# create model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_epochs = 75 # NUM EPOCHS

model = CNN().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# eval accuracy within training loop
def eval_acc_loader(loader):
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    if was_training:
        model.train()
    return 100.0 * correct / total

# eval loss within training loop
def eval_loss_loader(loader):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total += bs
    if was_training:
        model.train()
    return total_loss / total

# training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    train_acc = eval_acc_loader(train_eval_loader)
    test_acc  = eval_acc_loader(test_loader)

    train_loss_eval = eval_loss_loader(train_eval_loader)
    test_loss_eval  = eval_loss_loader(test_loader)

    print("lr:", optimizer.param_groups[0]["lr"])
    print(f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    print(f"Train Loss(eval): {train_loss_eval:.4f} | Test Loss(eval): {test_loss_eval:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    print()
