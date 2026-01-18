import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset

#from layers.WHT import WHTConv2D

# download format
# turns MNIST images to PyTorch tensors and normalizes between [-1,1] centered at 0
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5), # can be 10
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

raw_test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,            # testing data so labels are unknown during training
    download=True,
    transform=test_transform
)

raw_train_subset = Subset(raw_train_dataset, range(20000))
raw_test_subset = Subset(raw_test_dataset, range(2000))

# use raw_train_small and raw_test_small to decrease number of training and test images
# use raw_train_dataset and raw_test_dataset for entire dataset
train_dataset = raw_train_dataset
test_dataset  = raw_test_dataset

# loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=128,         # each epoch is 128 samples
    shuffle=True            # randomize after each training epoch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,         # each epoch is 256 samples
    shuffle=False
)

# test shapes of pytorch datasets
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

# print("train size", len(train_loader.dataset))
# print("test size", len(test_loader.dataset))

# CNN model
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        #self.wht = WHTConv2D(height=14, width=14, in_channels=32, out_channels=32, pods=1, residual=True)

        self.conv1a = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)

        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)

        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv4a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4b = nn.Conv2d(256, 256, 3, padding=1)

        # a and b batch norms for their corresponding conv layer because batchnorm learns
        self.bn1a = nn.BatchNorm2d(32)
        self.bn2a = nn.BatchNorm2d(64)
        self.bn3a = nn.BatchNorm2d(128)
        self.bn4a = nn.BatchNorm2d(256)

        self.bn1b = nn.BatchNorm2d(32)
        self.bn2b = nn.BatchNorm2d(64)
        self.bn3b = nn.BatchNorm2d(128)
        self.bn4b = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        #self.drop = nn.Dropout(0.0)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 10)
        #self.fc2 = nn.Linear(128, 10)
        #self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
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
        x = self.fc1(x)
        #x = self.drop(x)
        #x = self.fc2(x)

        #x = self.drop(x)
        #x = self.fc3(x)

        return x

# create model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 50 # NUM EPOCHS

model = CNN().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# eval accuracy within training loop
def eval_acc():
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    model.train()
    return 100 * correct / total

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
    print(f"Test Accuracy: {eval_acc():.2f}%")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# get accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")