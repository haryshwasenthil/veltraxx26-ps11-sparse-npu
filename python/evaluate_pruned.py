import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


# ============================================================
# 1. Define the same MLP architecture
# ============================================================

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


# ============================================================
# 2. Locate pruned model
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "results",
    "pruned_mlp.pth"
)

print("Loading pruned model:")
print(MODEL_PATH)


# ============================================================
# 3. Load model
# ============================================================

model = MLP()

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()

print("Pruned model loaded successfully!")


# ============================================================
# 4. Load MNIST test dataset
# ============================================================

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root=os.path.join(ROOT_DIR, "data"),
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ============================================================
# 5. Test accuracy
# ============================================================

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

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


# ============================================================
# 6. Display result
# ============================================================

baseline_accuracy = 97.41

accuracy_drop = baseline_accuracy - accuracy

print()
print("========================================")
print("PRUNED MODEL ACCURACY")
print("========================================")

print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
print(f"Pruned accuracy:   {accuracy:.2f}%")
print(f"Accuracy change:   {accuracy_drop:.2f}%")

print("========================================")