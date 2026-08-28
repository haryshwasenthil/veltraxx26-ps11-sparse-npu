import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EXPORT_DIR = os.path.join(BASE_DIR, "cpp", "data")

os.makedirs(EXPORT_DIR, exist_ok=True)


def export_array(npy_file, txt_file):
    data = np.load(
        os.path.join(RESULTS_DIR, npy_file)
    )

    np.savetxt(
        os.path.join(EXPORT_DIR, txt_file),
        data,
        fmt="%.8g"
    )

    print(f"Exported: {txt_file}")


layers = ["fc1", "fc2", "fc3"]

for layer in layers:

    export_array(
        f"{layer}_values.npy",
        f"{layer}_values.txt"
    )

    export_array(
        f"{layer}_column_indices.npy",
        f"{layer}_column_indices.txt"
    )

    export_array(
        f"{layer}_row_ptr.npy",
        f"{layer}_row_ptr.txt"
    )


print("\n========================================")
print("CSR EXPORT COMPLETE")
print("========================================")

print("C++ data location:")
print(EXPORT_DIR)