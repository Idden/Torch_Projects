import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

class HadamardTransform:
    def __init__(self):
        self.backend = AerSimulator(method="statevector")

    def __call__(self, x):

        # ensures cpu usage
        if x.is_cuda:
            x = x.cpu()

        # flatten image for qiskit
        img_flat = x.flatten().detach().numpy().astype(np.float64)
        len = img_flat.size

        # find number of qubits
        n_qubits = int(np.ceil(np.log2(len)))
        N = np.pow(2, n_qubits)

        # pad
        state_vector = np.zeros(N, dtype=np.float64)
        state_vector[:len] = img_flat

        # normalize constant
        norm = np.linalg.norm(state_vector)

        # cover edge case of fully black image
        if norm == 0:
            return torch.zeros_like(x)
        
        # normalize the state vector
        state_vector = state_vector / norm

        # circuit initialized with flattened image and hadamard on all rows
        qc = QuantumCircuit(n_qubits)
        qc.initialize(state_vector, qc.qubits)
        qc.h(range(n_qubits))
        qc.save_statevector()

        # create simulatible circuit
        tqc = transpile(qc, self.backend)
        result = self.backend.run(tqc).result()
        state = np.asarray(result.get_statevector(tqc))

        # take first 784 entries back, convert to real features
        y = np.real(state[:len]) * norm
        y = y.reshape(x.shape).astype(np.float32)

        return torch.from_numpy(y)

# download format
# turns MNIST images to PyTorch tensors and normalizes between [-1,1] centered at 0
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    HadamardTransform()
])

# download data
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,            # testing data so labels are unknown during training
    download=True,
    transform=transform
)

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

# CNN model
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# create model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()        # reset gradients
        outputs = model(images)      # forward
        loss = criterion(outputs, labels)
        loss.backward()              # backward
        optimizer.step()             # update weights

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# evalulate results and accuracy
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