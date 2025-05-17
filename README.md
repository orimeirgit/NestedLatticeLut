# C package for nested-lattice LUT

C library for compact matrix storage and accelerated vector operations using nested lattice quantization with lookup tables.

## Overview

This library implements nested lattice quantization for encoding vectors and matrices into a compact format. The key benefits are:

1. **Reduced Memory Footprint**: Encodes high-dimensional vectors using only a 4.5 bits per dimension, achieving a compression ratio of 7.11:1 compared to float32 representation.

2. **Accelerated Computations**: Uses precomputed lookup tables for inner products. When memory bandwidth is the bottleneck, we observe speedups of approximately 3x for vector operations. Platform-specific benchmarks show:
    - Apple M1 Pro CPU: ~3.4x speedup
    - Intel Core i5-12500H: ~2.8x speedup

The mean square error of our approach is approximately MSE ≈ 2^(-7) * (||A||²_F * ||B||²_F) / n, where ||X||_F  is the Frobenius norm, and n is the dimension.


## How It Works

The encoding process uses a nested lattice approach with the D4 lattice, where:
- Normalization and Hadamard transform is applied to the input vector.
- Each vector is divided into 4-dimensional subvectors
- Each subvector is encoded using lattice quantization with adaptive scaling (β values)
- Inner products are computed using precomputed lookup tables instead of direct multiplication

The optimal β values used in the quantization process were determined through simulations, which can be found in the simulation directory of this project.

## Project Contents

The project includes the following components:

- **Encoder**: Vector and matrix encoding functionality
- **Decoder**: Vector and matrix decoding functionality
- **LUT**: Lookup table generation and accelerated inner product computation
- **Matrix Operations**: Fast matrix multiplication using encoded representations
- **Utilities**: Helper functions for vector/matrix operations
- **Benchmarking**: Tools to evaluate performance and accuracy
- **Simulations**: A separate directory with its own README containing simulations used to determine optimal beta values for the quantization process

## Public API

### Encoding

```c
/**
 * @brief Encode a vector using nested-lattice quantization
 *
 * @param vector Input vector to encode
 * @param vector_length Length of the input vector
 * @param encoded_vectors Output array of size `vector_length` / 16, to store the coencoding result.
        each group contains four 4-dimension encoded subvectors (with b_0, b_1 ∈ [4]^4 for each subvector).
 * @param norm Output parameter to store the L2 norm of the vector (set to NULL if not needed)
 * @return true if encoding succeeded, false otherwise (overload error)
 */
int encode_float_vector(const float* vector, int vector_length,
                      EncodedVectorGroup** encoded_vectors, float* norm);

/**
 * Encode a matrix using nested-lattice quantization
 *
 * @param matrix Input matrix to encode (row-major format)
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @param encoded_matrices Output array of encoded vectors arrays, each of size `rows` / 16.
 * @param row_norms Output array to store the L2 norm of each row (set to NULL if not needed)
 * @return int Returns 1 if successful, 0 if failed
 */
int encode_float_matrix(const float* matrix, int rows, int cols,
                      EncodedVectorGroup*** encoded_matrices, float* row_norms);
```

### Decoding

```c
/**
 * Decode an encoded vector back to float format
 *
 * @param encoded_vectors Array of encoded vector groups
 * @param num_groups Number of encoded vector groups in the array
 * @param result Output float vector to store the reconstructed data
 * @param vector_length Length of the output vector
 * @param norm Vector norm to scale the output by (pass 0.0f to skip normalization)
 * @return int Returns 1 if successful, 0 if failed
 */
int decode_to_float_vector(EncodedVectorGroup* encoded_vectors, int num_groups, 
                          float* result, int vector_length, float norm);

/**
 * Decode an encoded matrix back to float format
 *
 * @param encoded_matrices Array of encoded vector groups representing a matrix
 * @param rows Number of rows in the matrix
 * @param cols Number of columns in the matrix
 * @param result Output float matrix to store the reconstructed data (row-major format)
 * @param row_norms Array of row norms to scale each row by (pass NULL to skip normalization)
 * @return int Returns 1 if successful, 0 if failed
 */
int decode_to_float_matrix(EncodedVectorGroup** encoded_matrices, int rows, 
                          int cols, float* result, const float* row_norms);
```

### Matrix Operations

```c
/**
 * Multiply two encoded matrices A and B using LUT-based computation
 * 
 * @param encoded_matrix_a First matrix in encoded form (rows_a x cols_a)
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A
 * @param encoded_matrix_b Second matrix in encoded form (rows_b x cols_b)
 * @param rows_b Number of rows in matrix B
 * @param cols_b Number of columns in matrix B
 * @param result Output matrix C (dimensions rows_a x cols_b) in row-major format
 * @param lut Lookup table for inner products. Use `init_inner_product_lut()` to initialize
 * @param row_norms_a Array of norms for each row in matrix A (may be NULL)
 * @param row_norms_b Array of norms for each row in matrix B (may be NULL)
 * @return int Returns 1 if successful, 0 if failed
 *
 * Note: The function expects cols_a == cols_b (the shared dimension for multiplication)
 */
int multiply_encoded_matrices(
    const EncodedVectorGroup** encoded_matrix_a, int rows_a, int cols_a,
    const EncodedVectorGroup** encoded_matrix_b, int rows_b, int cols_b,
    float* result, int8_t* lut, const float* row_norms_a, const float* row_norms_b);
```


### Lookup table

```c
/**
* Initialize the inner product table
*
* Pre-computes inner products for all possible pairs of encoded vectors
* and stores them in the table.
*
* @return the lut if initialization succeeded, false otherwise
  */
  int8_t* init_inner_product_lut();
```


## Benchmarking

The project includes benchmark utilities to measure the performance and accuracy of the nested lattice quantization approach. Follow these steps to run the benchmarks:

### 1. Generate Test Vectors

First, use the Python script to generate test vector pairs:

```bash
cd src
python3 generate_vectors.py my_vectors.bin 5000 2048 0.0 1.0
```

Arguments:
- `output_file`: Name of the binary file to create (default: vector_pairs.bin)
- `num_pairs`: Number of vector pairs to generate (default: 1000)
- `vector_length`: Length of each vector (default: 1024)
- `mean`: Mean of the normal distribution (default: 0.0)
- `stddev`: Standard deviation of the normal distribution (default: 1.0)

For help, run:
```bash
python3 generate_vectors.py --help
```

### 2. Build the Benchmark Executable

```bash
mkdir -p build
cd build
cmake ../src
make
```

### 3. Run the Benchmark

```bash
./bin/BenchmarkLUT my_vectors.bin my_results.csv 1
```

Command line arguments:
- `input_file`: Input binary file with vector pairs (default: vector_pairs.bin)
- `output_file`: Output CSV file for benchmark results (default: benchmark_results.csv)

For help, run:
```bash
./bin/BenchmarkLUT --help
```

## Requirements

- C99-compatible compiler
- Standard C library
- CMake (3.10 or higher) for building
- Python 3.x (for generating test vectors)


## References

- [Kaplan, Iris, and Ordentlich, Or. "High-Rate Nested-Lattice Quantized Matrix Multiplication with Small Lookup Tables" arXiv preprint arXiv:2505.13164 (2025).](https://arxiv.org/abs/2505.13164)
- [Ordentlich, Or, and Yury Polyanskiy. "Optimal Quantization for Matrix Multiplication." arXiv preprint arXiv:2410.13780 (2024).](https://arxiv.org/abs/2410.13780)



