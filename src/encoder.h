#ifndef NESTEDLATTICLUT_ENCODER_H
#define NESTEDLATTICLUT_ENCODER_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "encoded_vectors.h"

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
 * @brief Encode a matrix using nested-lattice quantization
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

#endif