#include "lut.h"
#include "utility.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


static int8_t int8_inner_product(const int8_t* x, const int8_t* y, int dimension) {
    int8_t result = 0;
    for (int i = 0; i < dimension; i++) {
        result += x[i] * y[i];
    }
    return result;
}


void cast_float_to_int16(const float* x, int8_t* result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        result[i] = (int8_t)roundf(x[i] + 0.001f);
    }
}

/**
 * Decode 8-bit encoded vector (4 dimensions with 2 bits each) to a reconstructed vector
 * This is a simplified version that just handles a single layer to build the LUT
 */
static void decode_single_layer(const uint8_t b_vector, int8_t* result) {
    float* G_b = (float*)malloc(D * sizeof(float));
    float* Q_L_G_b_div_q = (float*)malloc(D * sizeof(float));
    float* q_Q_L_G_b_div_q = (float*)malloc(D * sizeof(float));
    float* b = (float*)malloc(D * sizeof(float));
    float* G_b_div_q = (float*)malloc(D * sizeof(float));
    float* float_result = (float*)malloc(D * sizeof(float));

    
    if (!G_b || !Q_L_G_b_div_q || !q_Q_L_G_b_div_q || !b || !G_b_div_q) {
        free(G_b);
        free(Q_L_G_b_div_q);
        free(q_Q_L_G_b_div_q);
        free(b);
        free(G_b_div_q);
        return;
    }
    
    // Extract individual 2-bit values from the encoded vector
    for (int i = 0; i < D; i++) {
        b[i] = (float)((b_vector >> (i * 2)) & 0x03);
    }
    matrix_vector_multiply(d4_generating_matrix, b, G_b, D, D);
    vector_divide(G_b, G_b_div_q, D, (float)Q);
    d4_nearest_lattice_point(G_b_div_q, Q_L_G_b_div_q);
    vector_multiply(Q_L_G_b_div_q, q_Q_L_G_b_div_q, D, (float)Q);
    float_vector_subtract(G_b, q_Q_L_G_b_div_q, float_result, D);
    cast_float_to_int16(float_result, result, D);

    free(G_b);
    free(Q_L_G_b_div_q);
    free(q_Q_L_G_b_div_q);
    free(b);
    free(G_b_div_q);
}

uint16_t compute_lut_index(uint8_t b1, uint8_t b2) {
    return ((uint32_t)b1 << 8) | b2;
}

int8_t* init_inner_product_lut() {

    printf("Initializing inner product LUT with %u entries...\n", LUT_SIZE);
    int8_t* lut = (int8_t*)malloc(LUT_SIZE * sizeof(int8_t));
    int8_t* vector1 = (int8_t*)malloc(D * sizeof(int8_t));
    int8_t* vector2 = (int8_t*)malloc(D * sizeof(int8_t));
    
    if (!vector1 || !vector2 || !lut) {
        free(vector1);
        free(vector2);
        free(lut);
        return NULL;
    }
    
    for (uint8_t b1 = 0; b1 <= 255; b1++) {
        for (uint8_t b2 = 0; b2 <= 255; b2++) {
            decode_single_layer(b1, vector1);
            decode_single_layer(b2, vector2);

            const int8_t ip = int8_inner_product(vector1, vector2, D);
            const uint16_t index = compute_lut_index(b1, b2);
            lut[index] = ip;

            if (b2 == 255) {
                break;
            }
        }

        if (b1 == 255) {
            break;
        }
    }
    
    printf("LUT initialization completed.\n");
    
    free(vector1);
    free(vector2);
    return lut;
}

float lut_inner_product_for_vector_of_groups(const EncodedVectorGroup** vectors_a, const EncodedVectorGroup** vectors_b,
                                    const int8_t* lut, int idx, int vector_length, const uint16_t* beta_mult_values,
                                    const float beta_factor_2, float norm_a, float norm_b) {
    int total_result = 0;
    int num_chunks = (vector_length + ENCODED_VECTOR_DIM - 1) / ENCODED_VECTOR_DIM;

    const EncodedVectorGroup* vector_a = vectors_a[idx];
    const EncodedVectorGroup* vector_b = vectors_b[idx];

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        EncodedVectorGroup chunk_a = vector_a[chunk];
        EncodedVectorGroup chunk_b = vector_b[chunk];

        uint16_t beta_values_mult = beta_mult_values[(chunk_a.betaIndices & 0x03) << 2 |
                                                     chunk_b.betaIndices & 0x03];

        total_result += (
         lut[chunk_a.b_vectors[0] << 8 | chunk_b.b_vectors[0]] +
         (lut[chunk_a.b_vectors[0] << 8 | chunk_b.b_vectors[1]] << 2) +
         (lut[chunk_a.b_vectors[1] << 8 | chunk_b.b_vectors[0]] << 2) +
         (lut[chunk_a.b_vectors[1] << 8 | chunk_b.b_vectors[1]] << 4)
        ) * beta_values_mult;

        beta_values_mult = beta_mult_values[(chunk_a.betaIndices >> 2 & 0x03) << 2 |
                                            chunk_b.betaIndices >> 2 & 0x03];

        total_result += (
            lut[chunk_a.b_vectors[2] << 8 | chunk_b.b_vectors[2]] +
            (lut[chunk_a.b_vectors[2] << 8 | chunk_b.b_vectors[3]] << 2) +
            (lut[chunk_a.b_vectors[3] << 8 | chunk_b.b_vectors[2]] << 2) +
            (lut[chunk_a.b_vectors[3] << 8 | chunk_b.b_vectors[3]] << 4)
        ) * beta_values_mult;

        beta_values_mult = beta_mult_values[(chunk_a.betaIndices >> 4 & 0x03) << 2 |
                                            chunk_b.betaIndices >> 4 & 0x03];

        total_result += (
            lut[chunk_a.b_vectors[4] << 8 | chunk_b.b_vectors[4]] +
            (lut[chunk_a.b_vectors[4] << 8 | chunk_b.b_vectors[5]] << 2) +
            (lut[chunk_a.b_vectors[5] << 8 | chunk_b.b_vectors[4]] << 2) +
            (lut[chunk_a.b_vectors[5] << 8 | chunk_b.b_vectors[5]] << 4)
        ) * beta_values_mult;

        beta_values_mult = beta_mult_values[(chunk_a.betaIndices >> 6 & 0x03) << 2 |
                                            chunk_b.betaIndices >> 6 & 0x03];

        total_result += (
            lut[chunk_a.b_vectors[6] << 8 | chunk_b.b_vectors[6]] +
            (lut[chunk_a.b_vectors[6] << 8 | chunk_b.b_vectors[7]] << 2) +
            (lut[chunk_a.b_vectors[7] << 8 | chunk_b.b_vectors[6]] << 2) +
            (lut[chunk_a.b_vectors[7] << 8 | chunk_b.b_vectors[7]] << 4)
        ) * beta_values_mult;
    }

    // Start with the basic LUT-based inner product result
    float result = (float) total_result / beta_factor_2;
    
    // Apply normalization correction if norms are provided (multiply by ||a||*||b||/n)
    if (norm_a > 1e-10f && norm_b > 1e-10f) {
        result = result * (norm_a * norm_b) / (float)vector_length;
    }
    
    return result;
}
