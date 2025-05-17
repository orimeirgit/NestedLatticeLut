#ifndef NESTEDLATTICLUT_DECODER_H
#define NESTEDLATTICLUT_DECODER_H

#include <stdint.h>
#include "encoder.h"


/**
* @brief Decodes an array of encoded vector groups back to a float vector
*
* @param encoded_vectors Array of encoded vector groups
* @param num_groups Number of encoded vector groups in the array
* @param result Output float vector to store the reconstructed data
* @param vector_length Length of the output vector
* @param norm Vector norm to scale the output by (pass 0.0f to skip normalization)
* @return int Returns 1 if successful, 0 if failed
*/
int decode_to_float_vector(EncodedVectorGroup* encoded_vectors, int num_groups, float* result, int vector_length,
                           float norm);

/**
* @brief Decodes an array of encoded matrix groups back to a float matrix
*
* @param encoded_matrices Array of encoded vector groups representing a matrix
* @param rows Number of rows in the matrix
* @param cols Number of columns in the matrix
* @param result Output float matrix to store the reconstructed data (row-major format)
* @param row_norms Array of row norms to scale each row by (pass NULL to skip normalization)
* @return int Returns 1 if successful, 0 if failed
*/
int decode_to_float_matrix(EncodedVectorGroup** encoded_matrices, int rows, int cols, float* result,
                           const float* row_norms);

#endif //NESTEDLATTICLUT_DECODER_H


