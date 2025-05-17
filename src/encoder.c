#include <string.h>
#include <math.h>

#include "encoder.h"
#include "utility.h"
#include "encoded_vectors.h"


bool encode_vector(const float* x, uint8_t** encoded_vectors) {
    float* g_tilde = (float*)malloc(D * sizeof(float));
    float* nearest_lattice_point_vector = (float*)malloc(D * sizeof(float));
    float* temp = (float*)malloc(D * sizeof(float));
    int* rounded_temp = (int*)malloc(D * sizeof(int));
    
    if (!g_tilde || !nearest_lattice_point_vector || !temp || !rounded_temp) {
        free(g_tilde);
        free(nearest_lattice_point_vector);
        free(temp);
        free(rounded_temp);
        return false;
    }

    memcpy(g_tilde, x, D * sizeof(float));

    for (int m = 0; m < M; m++) {
        d4_nearest_lattice_point(g_tilde, nearest_lattice_point_vector);
        matrix_vector_multiply(d4_inverted_matrix, nearest_lattice_point_vector, temp, D, D);
        round_vector_to_int(temp, rounded_temp, D);
        vector_mod_q(rounded_temp, encoded_vectors[m], D, Q);
        vector_divide(nearest_lattice_point_vector, g_tilde, D, (float)Q);
    }

    bool overload_error = false;
    d4_nearest_lattice_point(g_tilde, temp);

    for (int i = 0; i < D; i++) {
        if ((float) fabsf(temp[i]) > 1e-6) {
            overload_error = true;
            break;
        }
    }

    free(g_tilde);
    free(nearest_lattice_point_vector);
    free(temp);
    free(rounded_temp);

    return !overload_error;
}

bool encode_vector_with_scaling(const float* x, const float beta, const float* dither,
                              uint8_t** encoded_vectors) {
    if (!x || !encoded_vectors || beta <= 0.0f) {
        return false;
    }

    ScalingDithering sd;
    if (!init_scaling_dithering(&sd, beta, dither, D)) {
        return false;
    }
    
    float* preprocessed_x = (float*)malloc(D * sizeof(float));
    if (!preprocessed_x) {
        free_scaling_dithering(&sd);
        return false;
    }
    
    apply_pre_encoding(x, &sd, preprocessed_x, D);
    
    const bool success = encode_vector(preprocessed_x, encoded_vectors);
    
    free(preprocessed_x);
    free_scaling_dithering(&sd);
    
    return success;
}


bool encode_vector_in_D4(const float x[4], EncodedVector* encoded,
    const uint8_t beta_index, const float* dither) {
    uint8_t** uint_encoded_vectors = (uint8_t**)malloc(M * Q * sizeof(uint32_t));
    for (int m = 0; m < M; m++) {
        uint_encoded_vectors[m] = (uint8_t*)malloc(D * sizeof(uint32_t));
        if (!uint_encoded_vectors[m]) {
            printf("Memory allocation failed!\n");
            for (int i = 0; i < m; i++) {
                free(uint_encoded_vectors[i]);
            }
            free(uint_encoded_vectors);
            return 1;
        }
    }

    const float beta = (float)get_D4_beta_for_index(beta_index);

    const bool success = encode_vector_with_scaling(x, beta, dither, uint_encoded_vectors);

    if (!success) {
        for (int m = 0; m < M; m++) {
            free(uint_encoded_vectors[m]);
        }
        free(uint_encoded_vectors);
        return false;
    }

    for (int i=0; i < 4; i++) {
        setEncodedVector(encoded, uint_encoded_vectors);
    }

    for (int m = 0; m < M; m++) {
        free(uint_encoded_vectors[m]);
    }
    free(uint_encoded_vectors);

    return true;
}

/**
 * Find the best beta value for encoding (smallest beta with no overload)
 */
bool find_best_beta_for_D4(const float* x, EncodedVector* b_vectors,
    const float* dither, uint8_t* found_beta_index) {
    
    for (uint8_t beta_idx = 0; beta_idx < 4; beta_idx++) {
        if (encode_vector_in_D4(x, b_vectors, beta_idx, dither)) {
            *found_beta_index = beta_idx;
            return true;
        }
    }
    
    return false;
}


bool find_closest_encodable_vector_D4(const float* x,
    EncodedVector* b_vectors, const float* dither, float* scaled_vector,
    uint8_t* beta_index) {
    
    float direction[D];
    vector_normalize(x, direction, D);
    vector_multiply(direction, scaled_vector, D, 3.1f * 500 );

    return find_best_beta_for_D4(scaled_vector, b_vectors, dither, beta_index);
}


bool find_best_betas_for_group_and_encode(const float* x, EncodedVectorGroup* vector_group,
                                     const float* dither) {
    memset(vector_group, 0, sizeof(EncodedVectorGroup));

    uint8_t*** values = (uint8_t***)malloc(4 * sizeof(uint8_t**));
    uint8_t* beta_indices = (uint8_t*)malloc(4 * sizeof(uint8_t));

    if (!values || !beta_indices) {
        if (values) free(values);
        if (beta_indices) free(beta_indices);
        return false;
    }

    // Initialize subvector storage
    for (int i = 0; i < 4; i++) {
        values[i] = (uint8_t**)malloc(M * sizeof(uint8_t*));
        if (!values[i]) {
            for (int j = 0; j < i; j++) {
                for (int k = 0; k < M; k++) {
                    free(values[j][k]);
                }
                free(values[j]);
            }
            free(values);
            free(beta_indices);
            return false;
        }

        for (int j = 0; j < M; j++) {
            values[i][j] = (uint8_t*)malloc(D * sizeof(uint8_t));
            if (!values[i][j]) {
                for (int k = 0; k < i; k++) {
                    for (int l = 0; l < M; l++) {
                        free(values[k][l]);
                    }
                    free(values[k]);
                }
                for (int l = 0; l < j; l++) {
                    free(values[i][l]);
                }
                free(values[i]);
                free(values);
                free(beta_indices);
                return false;
            }
        }
    }

    int success_count = 0;

    for (int i = 0; i < 4; i++) {


        float subvector[D];
        float* closest_encodable = malloc(D * sizeof(float));

        for (int j = 0; j < D; j++) {
            subvector[j] = x[i * D + j];
        }

        // Find the best beta for this subvector
        EncodedVector encoded;
        if (find_best_beta_for_D4(subvector, &encoded, dither, &beta_indices[i])) {
            success_count++;
            getValuesFromEncodedVector(&encoded, values[i]);
        } else {

            bool success = find_closest_encodable_vector_D4(
                subvector, &encoded,
                NULL, closest_encodable, &(beta_indices[i])
            );

            if (success) {
                getValuesFromEncodedVector(&encoded, values[i]);

            } else {
                memset(&encoded, 0, sizeof(EncodedVector));
            }
        }

        free(closest_encodable);
    }

    setEncodedVectorGroup(vector_group, (const uint8_t***)values, beta_indices);

    // Free allocated memory
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < M; j++) {
            free(values[i][j]);
        }
        free(values[i]);
    }
    free(values);
    free(beta_indices);

    return (success_count > 0);
}

int encode_float_vector(const float* vector, int vector_length,  EncodedVectorGroup** encoded_vectors, float* norm) {
    int num_chunks = (vector_length + ENCODED_VECTOR_DIM - 1) / ENCODED_VECTOR_DIM;

    *encoded_vectors = (EncodedVectorGroup*)malloc(num_chunks * sizeof(EncodedVectorGroup));

    if (!*encoded_vectors) {
        printf("Error: Memory allocation failed\n");
        return 0;
    }

    float* transformed_vector = (float*)malloc(vector_length * sizeof(float));
    if (!transformed_vector) {
        printf("Error: Memory allocation failed\n");
        free(*encoded_vectors);
        *encoded_vectors = NULL;
        return 0;
    }

    // First calculate the norm of the vector if needed
    float vector_norm_value = 0.0f;
    if (norm != NULL) {
        vector_norm_value = vector_norm(vector, vector_length);
        *norm = vector_norm_value;
    }
    
    // Apply normalization factor (sqrt(n)/||a||) if norm is not near zero
    float normalization_factor = 1.0f;
    if (vector_norm_value > 1e-10f) {
        normalization_factor = sqrtf((float)vector_length) / vector_norm_value;
    }

    // First, apply normalization to the input vector
    float* normalized_vector = (float*)malloc(vector_length * sizeof(float));
    if (!normalized_vector) {
        printf("Error: Memory allocation failed\n");
        free(transformed_vector);
        free(*encoded_vectors);
        *encoded_vectors = NULL;
        return 0;
    }

    for (int i = 0; i < vector_length; i++) {
        normalized_vector[i] = vector[i] * normalization_factor;
    }

    // Then apply Hadamard transform
    apply_hadamard_transform(normalized_vector, transformed_vector, vector_length);
    free(normalized_vector);


    // Process each chunk
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        float chunk_vector[ENCODED_VECTOR_DIM];

        for (int i = 0; i < ENCODED_VECTOR_DIM; i++) {
            int idx = chunk * ENCODED_VECTOR_DIM + i;
            if (idx < vector_length) {
                chunk_vector[i] = transformed_vector[idx] * (float)BETA_FACTOR;
            } else {
                chunk_vector[i] = 0.0f;  // Pad with zeros if past the end
            }
        }

        bool success = find_best_betas_for_group_and_encode(chunk_vector, &(*encoded_vectors)[chunk], NULL);

        if (!success) {
            printf("Chunk %d: Encoding failed\n", chunk);
        }

        // Print progress periodically for large vectors
        if ((chunk % 1000 == 0 || chunk == num_chunks - 1) && num_chunks > 1000) {
            printf("Encoded %d/%d chunks\n", chunk + 1, num_chunks);
        }
    }

    free(transformed_vector);
    return 1;
}

int encode_float_matrix(const float* matrix, int rows, int cols,
                      EncodedVectorGroup*** encoded_matrices, float* row_norms) {
    *encoded_matrices = (EncodedVectorGroup**)malloc(rows * sizeof(EncodedVectorGroup*));
    if (!*encoded_matrices) {
        printf("Error: Memory allocation failed for encoded matrices\n");
        return 0;
    }

    for (int i = 0; i < rows; i++) {
        const float* row = matrix + (i * cols);
        
        float* row_norm_ptr = (row_norms != NULL) ? &row_norms[i] : NULL;
        
        if (!encode_float_vector(row, cols, &((*encoded_matrices)[i]), row_norm_ptr)) {
            for (int j = 0; j < i; j++) {
                free((*encoded_matrices)[j]);
            }
            free(*encoded_matrices);
            *encoded_matrices = NULL;
            printf("Error: Failed to encode row %d of matrix\n", i);
            return 0;
        }
        
        // Print progress for large matrices
        if ((i % 1000 == 0 || i == rows - 1) && rows > 1000) {
            printf("Encoded %d/%d rows\n", i + 1, rows);
        }
    }
    
    return 1;
}
