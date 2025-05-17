#ifndef NESTEDLATTICLUT_LUT_H
#define NESTEDLATTICLUT_LUT_H

#include <stdint.h>
#include <stdbool.h>
#include "encoded_vectors.h"
#include "encoder.h"

/**
 * @brief Initialize the inner product table
 * 
 * Pre-computes inner products for all possible pairs of encoded vectors
 * and stores them in the table.
 *
 * @return the lut if initialization succeeded, false otherwise
 */
int8_t* init_inner_product_lut();

/**
 * Performs inner product for a vector of groups using the LUT multiplication algorithm.
 * @param vectors_a First vector of encoded groups
 * @param vectors_b Second vector of encoded groups
 * @param lut The lookup table containing precomputed inner products
 * @param idx The index of the vector to use
 * @param vector_length Length of the vectors
 * @param beta_mult_values Beta multiplication values table
 * @param beta_factor_2 Beta factor squared (for normalization)
 * @param norm_a Norm of the first vector (pass 0.0f to skip normalization)
 * @param norm_b Norm of the second vector (pass 0.0f to skip normalization)
 * @return The inner product result, properly normalized if norms are provided
 */
float lut_inner_product_for_vector_of_groups(const EncodedVectorGroup** vectors_a, const EncodedVectorGroup** vectors_b,
                                             const int8_t* lut, int idx, int vector_length, const uint16_t* beta_mult_values,
                                             float beta_factor_2, float norm_a, float norm_b);

#endif // NESTEDLATTICLUT_LUT_H