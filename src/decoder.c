#include "decoder.h"
#include "utility.h"
#include "encoded_vectors.h"

#include <stdlib.h>
#include <math.h>

void decode_vector(const uint8_t** encoded_vectors, float* result) {
    for (int i = 0; i < D; i++) {
        result[i] = 0.0f;
    }
    
    float* G_bm = (float*)malloc(D * sizeof(float));
    float* G_bm_div_q = (float*)malloc(D * sizeof(float));
    float* Q_L_G_bm_div_q = (float*)malloc(D * sizeof(float));
    float* q_Q_L_G_bm_div_q = (float*)malloc(D * sizeof(float));
    float* xm = (float*)malloc(D * sizeof(float));
    float* qm_xm = (float*)malloc(D * sizeof(float));
    float* temp_result = (float*)malloc(D * sizeof(float));
    float* bm_float = (float*)malloc(D * sizeof(float));

    if (!G_bm || !G_bm_div_q || !Q_L_G_bm_div_q || !q_Q_L_G_bm_div_q || !xm || !qm_xm || !temp_result || !bm_float) {
        free(G_bm);
        free(G_bm_div_q);
        free(Q_L_G_bm_div_q);
        free(q_Q_L_G_bm_div_q);
        free(xm);
        free(qm_xm);
        free(temp_result);
        free(bm_float);
        return;
    }

    for (int m = 0; m < M; m++) {
        for (int i = 0; i < D; i++) {
            bm_float[i] = (float)encoded_vectors[m][i];
        }
        matrix_vector_multiply(d4_generating_matrix, bm_float, G_bm, D, D);
        vector_divide(G_bm, G_bm_div_q, D, (float)Q);
        d4_nearest_lattice_point(G_bm_div_q, Q_L_G_bm_div_q);
        vector_multiply(Q_L_G_bm_div_q, q_Q_L_G_bm_div_q, D, (float)Q);
        float_vector_subtract(G_bm, q_Q_L_G_bm_div_q, xm, D);
        const float qm = powf((float)Q, (float)m);
        vector_multiply(xm, qm_xm, D, qm);
        vector_add(result, qm_xm, temp_result, D);
        for (int i = 0; i < D; i++) {
            result[i] = temp_result[i];
        }
    }
    
    free(G_bm);
    free(G_bm_div_q);
    free(Q_L_G_bm_div_q);
    free(q_Q_L_G_bm_div_q);
    free(xm);
    free(qm_xm);
    free(temp_result);
    free(bm_float);
}

void decode_vector_with_scaling(const uint8_t** encoded_vectors,float beta, const float* dither, float* result) {
    if (!encoded_vectors || !result || beta <= 0.0f) {
        return;
    }

    ScalingDithering sd;
    if (!init_scaling_dithering(&sd, beta, dither, D)) {
        return;
    }
    
    float* decoded_x = (float*)malloc(D * sizeof(float));
    if (!decoded_x) {
        free_scaling_dithering(&sd);
        return;
    }
    
    decode_vector(encoded_vectors, decoded_x);

    apply_post_decoding(decoded_x, &sd, result, D);

    free(decoded_x);
    free_scaling_dithering(&sd);
}

void decode_vector_in_D4(const EncodedVector* encoded, float* result, uint8_t beta_index) {
    const float beta = (float)get_D4_beta_for_index(beta_index);
    const float dither[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint8_t** encoded_vector_array = (uint8_t**)malloc(M * sizeof(uint8_t*));
    for (int m = 0; m < M; m++) {
        encoded_vector_array[m] = (uint8_t*)malloc(D * sizeof(uint8_t));
        if (!encoded_vector_array[m]) {
            printf("Memory allocation failed!\n");
            for (int i = 0; i < m; i++) {
                free(encoded_vector_array[i]);
            }
            free(encoded_vector_array);
            return;
        }
    }

    getValuesFromEncodedVector(encoded, encoded_vector_array);
    decode_vector_with_scaling(encoded_vector_array, beta, dither, result);

    for (int m = 0; m < M; m++) {
        free(encoded_vector_array[m]);
    }
    free(encoded_vector_array);
}

int decode_to_float_vector(EncodedVectorGroup* encoded_vectors, int num_groups, float* result, int vector_length,
                           float norm) {
    if (!encoded_vectors || !result) {
        printf("Error: Invalid input parameters\n");
        return 0;
    }

    
    // Memory for extracting encoded values from a vector group
    uint8_t*** values = (uint8_t***)malloc(4 * sizeof(uint8_t**));
    uint8_t* beta_indices = (uint8_t*)malloc(4 * sizeof(uint8_t));

    if (!values || !beta_indices) {
        printf("Error: Memory allocation failed\n");
        if (values) free(values);
        if (beta_indices) free(beta_indices);
        return 0;
    }
    
    // Initialize the values array
    for (int i = 0; i < 4; i++) {
        values[i] = (uint8_t**)malloc(M * sizeof(uint8_t*));
        if (!values[i]) {
            printf("Error: Memory allocation failed\n");
            for (int j = 0; j < i; j++) {
                for (int k = 0; k < M; k++) {
                    free(values[j][k]);
                }
                free(values[j]);
            }
            free(values);
            free(beta_indices);
            return 0;
        }
        
        for (int j = 0; j < M; j++) {
            values[i][j] = (uint8_t*)malloc(D * sizeof(uint8_t));
            if (!values[i][j]) {
                printf("Error: Memory allocation failed\n");
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
                return 0;
            }
        }
    }
    
    // Decode each group
    for (int chunk = 0; chunk < num_groups; chunk++) {
        getValuesForEncodedVectorGroup(&encoded_vectors[chunk], values);
        getBetaIndicesForEncodedVectorGroup(&encoded_vectors[chunk], beta_indices);
        
        // Process each 4D subvector within the chunk
        for (int subvec = 0; subvec < 4; subvec++) {
            float subvector_result[D];
            EncodedVector encoded;
            
            setEncodedVector(&encoded, values[subvec]);
            
            decode_vector_in_D4(&encoded, subvector_result, beta_indices[subvec]);
            
            // Scale the result back and copy to the output array
            for (int d = 0; d < D; d++) {
                int idx = chunk * ENCODED_VECTOR_DIM + subvec * D + d;
                if (idx < vector_length) {
                    result[idx] = subvector_result[d] / (float)BETA_FACTOR;
                }
            }
        }
    }

    // Apply denormalization if norm is provided (multiply by ||a||/sqrt(n))
    if (norm > 1e-10f) {
        float denormalization_factor = norm / sqrtf((float)vector_length);
        for (int i = 0; i < vector_length; i++) {
            result[i] *= denormalization_factor;
        }
    }
    
    // Free allocated memory
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < M; j++) {
            free(values[i][j]);
        }
        free(values[i]);
    }
    free(values);
    free(beta_indices);
    
    return 1;
}

int decode_to_float_matrix(EncodedVectorGroup** encoded_matrices, int rows, int cols, float* result,
                           const float* row_norms) {
    if (!encoded_matrices || !result) {
        printf("Error: Invalid input parameters\n");
        return 0;
    }
    
    // Decode each row of the matrix
    for (int i = 0; i < rows; i++) {
        // Get pointer to the start of the current row in the result matrix
        float* row_result = result + (i * cols);
        
        // Calculate number of groups needed for this row
        int num_groups = (cols + ENCODED_VECTOR_DIM - 1) / ENCODED_VECTOR_DIM;
        
        // Get the norm for this row if provided
        float row_norm = (row_norms != NULL) ? row_norms[i] : 0.0f;
        
        // Decode this row
        if (!decode_to_float_vector(encoded_matrices[i], num_groups, row_result, cols, row_norm)) {
            printf("Error: Failed to decode row %d of matrix\n", i);
            return 0;
        }
        
        // Print progress for large matrices
        if ((i % 1000 == 0 || i == rows - 1) && rows > 1000) {
            printf("Decoded %d/%d rows\n", i + 1, rows);
        }
    }
    
    return 1;
}
