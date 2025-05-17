#pragma once
#include <stdint.h>

#define Q 4
#define M 2
#define D 4
#define ENCODED_VECTOR_DIM 16 // Every 16 elements are encoded in a group of 4 vectors.
#define BETA_FACTOR 500 // Scaling factor for all factor, to make the relevant beta values integers.
#define LUT_SIZE 65536 // 2^16 for 4D lattice with q=4, M=2, and 2 bits per dimension


/**
 * Struct representing a 16-dimensional encoded vector with q=4,M=2 and beta=2 bits
 */

typedef struct {
    uint8_t b_vectors[8];
    uint8_t betaIndices;
} EncodedVectorGroup;

typedef struct {
    uint8_t b_vectors[2]; // 4 dimensions * 2 bits = 8 bits = 1 byte. M * 1 byte = 2 bytes.
} EncodedVector;

extern const float d4_generating_matrix[16];
extern const float d4_inverted_matrix[16];

void setEncodedVector(EncodedVector* vector, const uint8_t** values);

void getValuesFromEncodedVector(const EncodedVector* vector, uint8_t** result);

void setEncodedVectorGroup(EncodedVectorGroup* vector_group, const uint8_t*** values, const uint8_t* betaIndices);

void getValuesForEncodedVectorGroup(const EncodedVectorGroup* vector_group, uint8_t*** result);

void getBetaIndicesForEncodedVectorGroup(const EncodedVectorGroup* vector_group, uint8_t* betaIndices);

uint16_t get_D4_beta_for_index(const uint8_t betaIndex);

void print_vector_group(const char* name, const EncodedVectorGroup* vec_group);