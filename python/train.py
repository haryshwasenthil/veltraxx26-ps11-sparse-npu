import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Use CPU
device = torch.device("cpu")

print("Using device:", device)


# Load MNIST
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))


# Create the MLP
class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = x.view(x.size(0), 784)

        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))

        x = self.fc3(x)

        return x


model = MLP().to(device)

print(model)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Train the model
epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {average_loss:.4f}"
    )


# Test the model
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


accuracy = 100 * correct / total


print()
print("==============================")
print("BASELINE RESULTS")
print("==============================")
print(f"Accuracy: {accuracy:.2f}%")
print("==============================")


# Save the trained model
torch.save(
    model.state_dict(),
    "../dense_mlp.pth"
)

print()
print("Model saved as: ../dense_mlp.pth")


# Save the results
with open("../baseline_results.txt", "w") as f:

    f.write("PS11 Dense MLP Baseline\n")
    f.write("=======================\n")
    f.write("Dataset: MNIST\n")
    f.write("Architecture: 784-128-64-10\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")

print("Results saved as: ../baseline_results.txt")