import numpy as np


# ============================================================
# MEMBER 2 - WEIGHT PRUNING + CSR COMPRESSION
# ============================================================

# ------------------------------------------------------------
# STEP 1: Create example neural-network weights
# ------------------------------------------------------------

weights = np.array([
    [0.5, 0.01, -0.8, 0.002],
    [0.7, -0.03, 0.001, 0.9],
    [0.02, 0.6, -0.004, 0.3]
], dtype=np.float32)

print("===== ORIGINAL WEIGHTS =====")
print(weights)


# ------------------------------------------------------------
# STEP 2: Set target sparsity
# ------------------------------------------------------------

target_sparsity = 0.60

print("\nTarget sparsity:", target_sparsity * 100, "%")


# ------------------------------------------------------------
# STEP 3: Flatten weights
# ------------------------------------------------------------

flat_weights = weights.flatten()

total_weights = len(flat_weights)

print("Total weights:", total_weights)


# ------------------------------------------------------------
# STEP 4: Decide how many weights to prune
# ------------------------------------------------------------

num_to_prune = int(
    np.ceil(total_weights * target_sparsity)
)

print("Weights to prune:", num_to_prune)


# ------------------------------------------------------------
# STEP 5: Calculate magnitude of every weight
# ------------------------------------------------------------

magnitudes = np.abs(flat_weights)


# ------------------------------------------------------------
# STEP 6: Find smallest weights
# ------------------------------------------------------------

prune_indices = np.argsort(magnitudes)[:num_to_prune]


# ------------------------------------------------------------
# STEP 7: Set selected weights to zero
# ------------------------------------------------------------

pruned_weights = weights.copy()

flat_pruned = pruned_weights.flatten()

flat_pruned[prune_indices] = 0

pruned_weights = flat_pruned.reshape(weights.shape)


print("\n===== PRUNED WEIGHTS =====")
print(pruned_weights)


# ------------------------------------------------------------
# STEP 8: Calculate actual sparsity
# ------------------------------------------------------------

total_weights = pruned_weights.size

zero_weights = np.count_nonzero(
    pruned_weights == 0
)

nonzero_weights = np.count_nonzero(
    pruned_weights
)

sparsity = (
    zero_weights / total_weights
) * 100


print("\n===== SPARSITY RESULTS =====")

print("Total weights:", total_weights)

print("Zero weights:", zero_weights)

print("Non-zero weights:", nonzero_weights)

print("Sparsity:", sparsity, "%")


# ------------------------------------------------------------
# STEP 9: Check whether we achieved 60% sparsity
# ------------------------------------------------------------

if sparsity >= 60:
    print("\nSUCCESS: Sparsity requirement >= 60%")
else:
    print("\nFAILED: Sparsity requirement < 60%")


# ------------------------------------------------------------
# STEP 10: CSR COMPRESSION
# ------------------------------------------------------------

values = []

column_indices = []

row_ptr = [0]


for row in pruned_weights:

    for col, value in enumerate(row):

        # Store only non-zero values
        if value != 0:

            values.append(value)

            column_indices.append(col)

    # Number of stored values so far
    row_ptr.append(len(values))


# ------------------------------------------------------------
# STEP 11: Display CSR data
# ------------------------------------------------------------

print("\n===== CSR DATA =====")

print("\nValues:")
print(values)

print("\nColumn indices:")
print(column_indices)

print("\nRow pointer:")
print(row_ptr)


# ------------------------------------------------------------
# STEP 12: Convert CSR arrays to proper data types
# ------------------------------------------------------------

values_array = np.array(
    values,
    dtype=np.float32
)

column_indices_array = np.array(
    column_indices,
    dtype=np.int32
)

row_ptr_array = np.array(
    row_ptr,
    dtype=np.int32
)


# ------------------------------------------------------------
# STEP 13: Save CSR files
# ------------------------------------------------------------

np.save(
    "values.npy",
    values_array
)

np.save(
    "column_indices.npy",
    column_indices_array
)

np.save(
    "row_ptr.npy",
    row_ptr_array
)


# ------------------------------------------------------------
# STEP 14: Final message
# ------------------------------------------------------------

print("\n===== FILES SAVED =====")

print("values.npy")
print("column_indices.npy")
print("row_ptr.npy")

print("\n===== MEMBER 2 PROTOTYPE COMPLETE =====")