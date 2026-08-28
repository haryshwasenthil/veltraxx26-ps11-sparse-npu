#include <iostream>
#include <vector>

using namespace std;

// CSR Sparse Matrix structure
struct CSRMatrix {
    int rows;
    int cols;

    vector<float> values;
    vector<int> col_indices;
    vector<int> row_ptr;
};

// Sparse Matrix × Vector multiplication
vector<float> sparseMatVec(
    const CSRMatrix& matrix,
    const vector<float>& input
) {
    vector<float> output(matrix.rows, 0.0f);

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

int main() {

    // Example sparse matrix:
    //
    // [ 0  5  0  2 ]
    // [ 1  0  0  0 ]
    // [ 0  0  3  4 ]

    CSRMatrix matrix;

    matrix.rows = 3;
    matrix.cols = 4;

    matrix.values = {5, 2, 1, 3, 4};

    matrix.col_indices = {1, 3, 0, 2, 3};

    matrix.row_ptr = {0, 2, 3, 5};

    vector<float> input = {10, 20, 30, 40};

    vector<float> output = sparseMatVec(matrix, input);

    cout << "Output: ";

    for (float value : output) {
        cout << value << " ";
    }

    cout << endl;

    return 0;
}