import torch
import torch.nn as nn
import numpy as np
import os


# ============================================================
# 1. Define the same MLP architecture used by Member 1
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
# 2. Load the trained model
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "dense_mlp.pth"
)

print("Loading model:")
print(MODEL_PATH)

model = MLP()

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()

print("Model loaded successfully!")


# ============================================================
# 3. Pruning function
# ============================================================

TARGET_SPARSITY = 0.60


def prune_matrix(weight_matrix, target_sparsity):

    flat = weight_matrix.flatten()

    total_weights = flat.numel()

    number_to_prune = int(
        np.ceil(total_weights * target_sparsity)
    )

    magnitudes = torch.abs(flat)

    prune_indices = torch.argsort(magnitudes)[:number_to_prune]

    pruned = flat.clone()

    pruned[prune_indices] = 0

    return pruned.reshape(weight_matrix.shape)


# ============================================================
# 4. Prune all MLP weight matrices
# ============================================================

print()
print("========================================")
print("PRUNING MLP")
print("========================================")

weight_names = [
    "fc1.weight",
    "fc2.weight",
    "fc3.weight"
]

total_weights = 0
total_zeros = 0

for name in weight_names:

    original = model.state_dict()[name].clone()

    pruned = prune_matrix(
        original,
        TARGET_SPARSITY
    )

    model.state_dict()[name].copy_(pruned)

    zeros = torch.sum(pruned == 0).item()
    weights = pruned.numel()

    sparsity = 100 * zeros / weights

    total_weights += weights
    total_zeros += zeros

    print()
    print(name)
    print("Shape:", tuple(pruned.shape))
    print("Total weights:", weights)
    print("Zero weights:", zeros)
    print(f"Sparsity: {sparsity:.2f}%")


# ============================================================
# 5. Overall sparsity
# ============================================================

overall_sparsity = (
    100 * total_zeros / total_weights
)

print()
print("========================================")
print("OVERALL SPARSITY")
print("========================================")

print("Total weights:", total_weights)
print("Total zeros:", total_zeros)
print("Non-zero weights:", total_weights - total_zeros)

print(
    f"Overall sparsity: {overall_sparsity:.2f}%"
)


if overall_sparsity >= 60:

    print("SUCCESS: Sparsity requirement >= 60%")

else:

    print("WARNING: Sparsity requirement not reached")


# ============================================================
# 6. CSR compression
# ============================================================

def matrix_to_csr(matrix):

    values = []
    column_indices = []
    row_ptr = [0]

    rows = matrix.shape[0]

    for row in range(rows):

        for col in range(matrix.shape[1]):

            value = matrix[row, col].item()

            if value != 0:

                values.append(value)
                column_indices.append(col)

        row_ptr.append(len(values))

    return (
        np.array(values, dtype=np.float32),
        np.array(column_indices, dtype=np.int32),
        np.array(row_ptr, dtype=np.int32)
    )


# ============================================================
# 7. Create output directory
# ============================================================

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 8. Compress each layer using CSR
# ============================================================

print()
print("========================================")
print("CSR COMPRESSION")
print("========================================")

for name in weight_names:

    matrix = model.state_dict()[name]

    values, column_indices, row_ptr = matrix_to_csr(
        matrix
    )

    layer_name = name.replace(".weight", "")

    values_path = os.path.join(
        RESULTS_DIR,
        layer_name + "_values.npy"
    )

    columns_path = os.path.join(
        RESULTS_DIR,
        layer_name + "_column_indices.npy"
    )

    row_ptr_path = os.path.join(
        RESULTS_DIR,
        layer_name + "_row_ptr.npy"
    )

    np.save(values_path, values)

    np.save(
        columns_path,
        column_indices
    )

    np.save(
        row_ptr_path,
        row_ptr
    )

    print()
    print(layer_name)

    print("Non-zero values:", len(values))

    print("Values file:", values_path)

    print(
        "Column indices file:",
        columns_path
    )

    print(
        "Row pointer file:",
        row_ptr_path
    )


# ============================================================
# 9. Save the pruned PyTorch model
# ============================================================

PRUNED_MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "pruned_mlp.pth"
)

torch.save(
    model.state_dict(),
    PRUNED_MODEL_PATH
)

print()
print("========================================")
print("PRUNING + CSR COMPLETE")
print("========================================")

print(
    "Pruned model saved:",
    PRUNED_MODEL_PATH
)

print()
print("Member 2 pipeline completed successfully!")