#ifndef SPARSE_ENGINE_H
#define SPARSE_ENGINE_H

#include <vector>

struct CSRMatrix {
    int rows;
    int cols;

    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

std::vector<float> sparseMatVec(
    const CSRMatrix& matrix,
    const std::vector<float>& input
);

#endif