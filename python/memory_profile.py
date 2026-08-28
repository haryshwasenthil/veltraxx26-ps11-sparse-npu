import numpy as np
import os


# ============================================================
# MEMBER 2 - CSR MEMORY PROFILING
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


# Layer information
layers = ["fc1", "fc2", "fc3"]


total_dense_bytes = 0
total_csr_bytes = 0


print("========================================")
print("CSR MEMORY PROFILING")
print("========================================")


for layer in layers:

    # --------------------------------------------------------
    # Load CSR files
    # --------------------------------------------------------

    values = np.load(
        os.path.join(
            RESULTS_DIR,
            f"{layer}_values.npy"
        )
    )

    column_indices = np.load(
        os.path.join(
            RESULTS_DIR,
            f"{layer}_column_indices.npy"
        )
    )

    row_ptr = np.load(
        os.path.join(
            RESULTS_DIR,
            f"{layer}_row_ptr.npy"
        )
    )


    # --------------------------------------------------------
    # Calculate number of elements
    # --------------------------------------------------------

    nonzero_count = len(values)

    # Dense weight count
    if layer == "fc1":
        rows = 128
        cols = 784

    elif layer == "fc2":
        rows = 64
        cols = 128

    else:
        rows = 10
        cols = 64


    dense_elements = rows * cols


    # --------------------------------------------------------
    # Dense memory
    # float32 = 4 bytes
    # --------------------------------------------------------

    dense_bytes = dense_elements * 4


    # --------------------------------------------------------
    # CSR memory
    #
    # values          = float32 = 4 bytes
    # column_indices  = int32   = 4 bytes
    # row_ptr         = int32   = 4 bytes
    # --------------------------------------------------------

    values_bytes = values.nbytes
    column_bytes = column_indices.nbytes
    row_ptr_bytes = row_ptr.nbytes

    csr_bytes = (
        values_bytes
        + column_bytes
        + row_ptr_bytes
    )


    total_dense_bytes += dense_bytes
    total_csr_bytes += csr_bytes


    # --------------------------------------------------------
    # Display layer result
    # --------------------------------------------------------

    print()
    print(layer)

    print("Dense memory:",
          dense_bytes,
          "bytes")

    print("CSR values:",
          values_bytes,
          "bytes")

    print("CSR column indices:",
          column_bytes,
          "bytes")

    print("CSR row pointer:",
          row_ptr_bytes,
          "bytes")

    print("Total CSR memory:",
          csr_bytes,
          "bytes")

    print(
        "Compression ratio:",
        round(dense_bytes / csr_bytes, 2),
        "x"
    )


# ============================================================
# Overall result
# ============================================================

print()
print("========================================")
print("OVERALL MEMORY RESULTS")
print("========================================")


print(
    "Dense memory:",
    total_dense_bytes,
    "bytes"
)

print(
    "CSR memory:",
    total_csr_bytes,
    "bytes"
)


compression_ratio = (
    total_dense_bytes /
    total_csr_bytes
)


memory_reduction = (
    1 -
    (total_csr_bytes / total_dense_bytes)
) * 100


print(
    "Compression ratio:",
    round(compression_ratio, 2),
    "x"
)

print(
    "Memory reduction:",
    round(memory_reduction, 2),
    "%"
)


print("========================================")


if compression_ratio >= 2:

    print(
        "SUCCESS: At least 2x memory reduction"
    )

else:

    print(
        "NOTE: CSR does not yet achieve 2x"
    )

print("========================================")