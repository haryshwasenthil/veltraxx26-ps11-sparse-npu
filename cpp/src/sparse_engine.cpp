#include "../include/sparse_engine.h"

std::vector<float> sparseMatVec(
    const CSRMatrix& matrix,
    const std::vector<float>& input
) {
    std::vector<float> output(matrix.rows, 0.0f);

    for (int row = 0; row < matrix.rows; row++) {

        for (int k = matrix.row_ptr[row];
             k < matrix.row_ptr[row + 1];
             k++) {

            output[row] +=
                matrix.values[k] *
                input[matrix.col_indices[k]];
        }
    }

    return output;
}