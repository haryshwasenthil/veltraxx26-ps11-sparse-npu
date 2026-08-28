#include "../include/sparse_engine.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace chrono;

// =============================================
// Dense Matrix-Vector Multiplication
// =============================================

vector<float> denseMatVec(
    const vector<float>& matrix,
    const vector<float>& input,
    int rows,
    int cols
) {
    vector<float> output(rows, 0.0f);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[i] += matrix[i * cols + j] * input[j];
        }
    }

    return output;
}


// =============================================
// Dense Memory Calculation
// =============================================

size_t denseMemory(int rows, int cols) {

    return static_cast<size_t>(rows)
           * cols
           * sizeof(float);
}


// =============================================
// CSR Memory Calculation
// =============================================

size_t csrMemory(const CSRMatrix& matrix) {

    return matrix.values.size()
           * sizeof(float)

           + matrix.col_indices.size()
           * sizeof(int)

           + matrix.row_ptr.size()
           * sizeof(int);
}


// =============================================
// Main Benchmark
// =============================================

int main() {

    // -----------------------------------------
    // Matrix Configuration
    // -----------------------------------------

    const int rows = 1000;
    const int cols = 1000;

    // Repeat operations for reliable timing
    const int iterations = 100;

    // 80% pruning / sparsity
    const float sparsity = 0.80f;


    // -----------------------------------------
    // Create Dense Matrix and Input Vector
    // -----------------------------------------

    vector<float> denseMatrix(
        rows * cols,
        0.0f
    );

    vector<float> input(cols);


    // Fixed seed gives reproducible results
    mt19937 generator(42);

    uniform_real_distribution<float>
        valueDist(-1.0f, 1.0f);

    uniform_real_distribution<float>
        probability(0.0f, 1.0f);


    // -----------------------------------------
    // Generate Input Vector
    // -----------------------------------------

    for (int j = 0; j < cols; j++) {

        input[j] =
            valueDist(generator);
    }


    // -----------------------------------------
    // Generate Matrix With 80% Zeros
    // -----------------------------------------

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < cols; j++) {

            // Only around 20% become non-zero
            if (probability(generator) > sparsity) {

                denseMatrix[i * cols + j] =
                    valueDist(generator);
            }
        }
    }


    // =============================================
    // Convert Dense Matrix to CSR Format
    // =============================================

    CSRMatrix sparseMatrix;

    sparseMatrix.rows = rows;
    sparseMatrix.cols = cols;

    // First row always starts at index 0
    sparseMatrix.row_ptr.push_back(0);


    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < cols; j++) {

            float value =
                denseMatrix[i * cols + j];

            // Store only non-zero values
            if (value != 0.0f) {

                sparseMatrix.values.push_back(
                    value
                );

                sparseMatrix.col_indices.push_back(
                    j
                );
            }
        }

        // Mark the end of the current row
        sparseMatrix.row_ptr.push_back(
            static_cast<int>(
                sparseMatrix.values.size()
            )
        );
    }


    // =============================================
    // Dense Execution Benchmark
    // =============================================

    vector<float> denseOutput;

    auto denseStart =
        high_resolution_clock::now();


    for (int i = 0;
         i < iterations;
         i++) {

        denseOutput =
            denseMatVec(
                denseMatrix,
                input,
                rows,
                cols
            );
    }


    auto denseEnd =
        high_resolution_clock::now();


    // =============================================
    // Sparse CSR Execution Benchmark
    // =============================================

    vector<float> sparseOutput;

    auto sparseStart =
        high_resolution_clock::now();


    for (int i = 0;
         i < iterations;
         i++) {

        sparseOutput =
            sparseMatVec(
                sparseMatrix,
                input
            );
    }


    auto sparseEnd =
        high_resolution_clock::now();


    // =============================================
    // Calculate Execution Time
    // =============================================

    double denseTime =

        duration<double, milli>(
            denseEnd - denseStart
        ).count();


    double sparseTime =

        duration<double, milli>(
            sparseEnd - sparseStart
        ).count();


    // =============================================
    // Memory Calculation
    // =============================================

    size_t denseMem =
        denseMemory(rows, cols);


    size_t sparseMem =
        csrMemory(sparseMatrix);


    // =============================================
    // Calculate Actual Sparsity
    // =============================================

    size_t totalElements =

        static_cast<size_t>(rows)
        * cols;


    size_t nonZero =

        sparseMatrix.values.size();


    double actualSparsity =

        100.0 *
        (
            1.0 -

            static_cast<double>(nonZero)
            / totalElements
        );


    // =============================================
    // Calculate Memory Reduction
    // =============================================

    double memoryReduction =

        static_cast<double>(denseMem)
        / sparseMem;


    // =============================================
    // Calculate Speedup
    // =============================================

    double speedup =

        denseTime
        / sparseTime;


    // =============================================
    // Verify Dense and Sparse Output
    // =============================================

    double maxDifference = 0.0;


    for (int i = 0;
         i < rows;
         i++) {

        double difference =

            abs(
                denseOutput[i]
                -
                sparseOutput[i]
            );


        if (difference > maxDifference) {

            maxDifference =
                difference;
        }
    }


    // =============================================
    // Display Results
    // =============================================

    cout << fixed
         << setprecision(2);


    cout << "\n";

    cout << "============================================\n";

    cout << "      PS11 SPARSE NPU BENCHMARK RESULTS\n";

    cout << "============================================\n\n";


    cout << "Matrix Size: "

         << rows
         << " x "
         << cols
         << "\n";


    cout << "Iterations: "

         << iterations
         << "\n";


    cout << "Non-Zero Values: "

         << nonZero
         << "\n";


    cout << "Sparsity: "

         << actualSparsity
         << "%\n\n";


    // Memory results

    cout << "----------- MEMORY COMPARISON -----------\n";

    cout << "Dense Memory: "

         << denseMem / 1024.0 / 1024.0
         << " MB\n";


    cout << "CSR Memory: "

         << sparseMem / 1024.0 / 1024.0
         << " MB\n";


    cout << "Memory Reduction: "

         << memoryReduction
         << "x\n\n";


    // Performance results

    cout << "--------- EXECUTION TIME COMPARISON ---------\n";

    cout << "Dense Execution Time: "

         << denseTime
         << " ms\n";


    cout << "Sparse Execution Time: "

         << sparseTime
         << " ms\n";


    cout << "Speedup: "

         << speedup
         << "x\n\n";


    // Verification

    cout << "--------------- VERIFICATION ---------------\n";

    cout << "Maximum Output Difference: "

         << maxDifference
         << "\n";


    if (maxDifference < 0.001) {

        cout << "Output Verification: PASSED\n";

    } else {

        cout << "Output Verification: FAILED\n";
    }


    // =============================================
    // PS11 Requirement Check
    // =============================================

    cout << "\n";

    cout << "------------- PS11 REQUIREMENTS ------------\n";


    cout << "Minimum 60% Sparsity: ";

    if (actualSparsity >= 60.0) {

        cout << "PASSED\n";

    } else {

        cout << "FAILED\n";
    }


    cout << "Minimum 2x Memory Reduction: ";

    if (memoryReduction >= 2.0) {

        cout << "PASSED\n";

    } else {

        cout << "FAILED\n";
    }


    cout << "Minimum 1.5x Speedup: ";

    if (speedup >= 1.5) {

        cout << "PASSED\n";

    } else {

        cout << "FAILED\n";
    }


    cout << "\n";

    cout << "============================================\n";

    return 0;
}