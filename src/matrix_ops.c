#include "matrix_ops.h"
#include "lut.h"
#include "utility.h"
#include "decoder.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/**
 * Performs multiplication of two encoded matrices using LUT-based computation
 * Assumes matrix B is already transposed (B_T)
 */
int multiply_encoded_matrices(
    const EncodedVectorGroup** encoded_matrix_a, int rows_a, int cols_a,
    const EncodedVectorGroup** encoded_matrix_b_T, int cols_b, int rows_b,
    float* result) {
    
    // For matrix multiplication A * B, we need cols_a == rows_b
    if (cols_a != rows_b) {
        printf("Error: Matrix dimensions mismatch for multiplication\n");
        printf("Matrix A: %d x %d, Matrix B: %d x %d\n", rows_a, cols_a, rows_b, cols_b);
        return 0;
    }
    
    // Check for valid input parameters
    if (!encoded_matrix_a || !encoded_matrix_b_T || !result) {
        printf("Error: Invalid input parameters\n");
        return 0;
    }
    
    // Initialize inner product LUT
    int8_t* lut = init_inner_product_lut();
    if (!lut) {
        printf("Error: Failed to initialize LUT\n");
        return 0;
    }
    
    // Prepare beta multiplication values table
    uint16_t beta_mult_values[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            beta_mult_values[i << 2 | j] = (uint16_t)(get_D4_beta_for_index(i) * get_D4_beta_for_index(j));
        }
    }
    
    // Calculate beta factor squared (for normalization)
    float beta_factor_2 = (float)BETA_FACTOR * (float)BETA_FACTOR;
    
    // Initialize result matrix with zeros
    memset(result, 0, rows_a * cols_b * sizeof(float));
    
    // Perform matrix multiplication C = A * B
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            // For each element C[i,j], compute inner product of row i from A and row j from B_T
            // Use the existing lut_inner_product_for_vector_of_groups function
            
            // Pass row i of A and row j of B_T (which is column j of B)
            const EncodedVectorGroup** row_a = &encoded_matrix_a[i];
            const EncodedVectorGroup** col_b = &encoded_matrix_b_T[j];
            
            // Compute the inner product
            float dot_product = lut_inner_product_for_vector_of_groups(
                row_a, col_b, lut, 0, cols_a, beta_mult_values, beta_factor_2
            );
            
            // Store the result
            result[i * cols_b + j] = dot_product;
        }
        
        // Print progress for large matrices
        if ((i % 100 == 0 || i == rows_a - 1) && rows_a > 100) {
            printf("Processed %d/%d rows of matrix multiplication\n", i + 1, rows_a);
        }
    }
    
    // Free LUT
    free(lut);
    
    return 1;
} 