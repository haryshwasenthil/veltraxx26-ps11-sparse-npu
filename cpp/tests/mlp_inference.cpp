#include "../include/sparse_engine.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


// Load float values from text file
std::vector<float> loadFloatFile(
    const std::string& filename
) {
    std::vector<float> data;
    std::ifstream file(filename);

    float value;

    while (file >> value) {
        data.push_back(value);
    }

    return data;
}


// Load integer values from text file
std::vector<int> loadIntFile(
    const std::string& filename
) {
    std::vector<int> data;
    std::ifstream file(filename);

    int value;

    while (file >> value) {
        data.push_back(value);
    }

    return data;
}


// Load one CSR layer
CSRMatrix loadCSRLayer(
    const std::string& layer,
    int rows,
    int cols
) {
    CSRMatrix matrix;

    matrix.rows = rows;
    matrix.cols = cols;

    matrix.values = loadFloatFile(
        "cpp/data/" + layer + "_values.txt"
    );

    matrix.col_indices = loadIntFile(
        "cpp/data/" + layer + "_column_indices.txt"
    );

    matrix.row_ptr = loadIntFile(
        "cpp/data/" + layer + "_row_ptr.txt"
    );

    return matrix;
}


// ReLU activation
void relu(std::vector<float>& values) {

    for (float& value : values) {

        if (value < 0.0f) {
            value = 0.0f;
        }
    }
}


// Find index of largest output
int argmax(
    const std::vector<float>& values
) {
    return std::max_element(
        values.begin(),
        values.end()
    ) - values.begin();
}


int main() {

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "   PS11 REAL SPARSE MLP INFERENCE\n";
    std::cout << "========================================\n\n";


    // Load actual CSR compressed MLP weights

    std::cout
        << "Loading CSR compressed MLP layers...\n";


    CSRMatrix fc1 = loadCSRLayer(
        "fc1",
        128,
        784
    );

    CSRMatrix fc2 = loadCSRLayer(
        "fc2",
        64,
        128
    );

    CSRMatrix fc3 = loadCSRLayer(
        "fc3",
        10,
        64
    );


    // Verify files loaded correctly

    if (
        fc1.values.empty() ||
        fc2.values.empty() ||
        fc3.values.empty()
    ) {

        std::cerr
            << "ERROR: CSR data files could not be loaded.\n";

        return 1;
    }


    std::cout
        << "FC1: "
        << fc1.values.size()
        << " non-zero weights\n";

    std::cout
        << "FC2: "
        << fc2.values.size()
        << " non-zero weights\n";

    std::cout
        << "FC3: "
        << fc3.values.size()
        << " non-zero weights\n\n";


    // Create a 784-element test input
    std::vector<float> input(
        784,
        0.0f
    );


    // Simple test pattern
    for (int i = 0; i < 784; i += 50) {

        input[i] = 1.0f;
    }


    std::cout
        << "Running sparse MLP inference...\n";


    // FC1: 784 -> 128
    std::vector<float> output1 =
        sparseMatVec(fc1, input);

    relu(output1);


    // FC2: 128 -> 64
    std::vector<float> output2 =
        sparseMatVec(fc2, output1);

    relu(output2);


    // FC3: 64 -> 10
    std::vector<float> output3 =
        sparseMatVec(fc3, output2);


    // Get prediction
    int prediction = argmax(output3);


    // Display results

    std::cout
        << "\nInference completed successfully!\n\n";


    std::cout << "Output scores:\n";

    for (int i = 0; i < 10; i++) {

        std::cout
            << "Class "
            << i
            << ": "
            << output3[i]
            << "\n";
    }


    std::cout
        << "\nPredicted Digit: "
        << prediction
        << "\n";


    std::cout
        << "\n========================================\n";

    std::cout
        << "SPARSE MLP EXECUTION: SUCCESS\n";

    std::cout
        << "========================================\n";


    return 0;
}