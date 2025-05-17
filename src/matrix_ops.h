#ifndef NESTEDLATTICLUT_MATRIX_OPS_H
#define NESTEDLATTICLUT_MATRIX_OPS_H

#include <stdint.h>
#include <stdbool.h>
#include "encoded_vectors.h"

/**
 * @brief Multiply two encoded matrices A and B using LUT-based computation
 * 
 * Matrix B must be provided in transposed form encoded (B_T) for computation.
 * 
 * @param encoded_matrix_a First encoded matrix A (dimensions m x k)
 * @param rows_a Number of rows in matrix A
 * @param cols_a Number of columns in matrix A
 * @param encoded_matrix_b_T Second encoded matrix B in TRANSPOSED form (dimensions n x k)
 * @param cols_b Number of columns in the original matrix B (rows in B_T)
 * @param rows_b Number of rows in the original matrix B (columns in B_T)
 * @param result Output matrix C (dimensions m x n) in row-major format
 * @return int Returns 1 if successful, 0 if failed
 */
int multiply_encoded_matrices(
    const EncodedVectorGroup** encoded_matrix_a, int rows_a, int cols_a,
    const EncodedVectorGroup** encoded_matrix_b_T, int cols_b, int rows_b,
    float* result);

#endif // NESTEDLATTICLUT_MATRIX_OPS_H 