# include "encoded_vectors.h"

# include <stdio.h>

const uint16_t BETA_VALUES[] = {56, 75, 99, 201};

const float d4_generating_matrix[16] = {
    -1.0f,  1.0f,  0.0f,  0.0f,
    -1.0f, -1.0f,  1.0f,  0.0f,
    0.0f,  0.0f, -1.0f,  1.0f,
    0.0f,  0.0f,  0.0f, -1.0f
};

const float d4_inverted_matrix[16] = {
    -0.5f, -0.5f, -0.5f, -0.5f,
    0.5f, -0.5f, -0.5f, -0.5f,
    -0.0f, -0.0f, -1.0f, -1.0f,
    -0.0f, -0.0f, -0.0f, -1.0f
};

void setEncodedVector(EncodedVector* vector, const uint8_t** values) {
    for (int i = 0; i < 2; i++) {
        vector->b_vectors[i] = 0;
        for (int j = 0; j < 4; j++) {
            // Each dimension uses 2 bits (beta=2)
            // Shift and OR the values into the correct position
            vector->b_vectors[i] |= (values[i][j] & 0x03) << (j * 2);
        }
    }
}

void setEncodedVectorGroup(EncodedVectorGroup* vector_group, const uint8_t*** values, const uint8_t* betaIndices) {
    for (int vectorIndex = 0; vectorIndex < 4; vectorIndex++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                // Each dimension uses 2 bits (beta=2)
                // Shift and OR the values into the correct position
                vector_group->b_vectors[vectorIndex * 2 + i] |= (values[vectorIndex][i][j] & 0x03) << (j * 2);
            }
        }

        // Set the beta index for the vector
        vector_group->betaIndices |= (betaIndices[vectorIndex] & 0x03) << (vectorIndex * 2);
    }
}

uint8_t getValueEncodedVector4DQ4Beta2(const EncodedVector* vector, uint8_t m, uint8_t dim) {
    // Extract 2 bits from the correct position
    return (vector->b_vectors[m] >> (dim * 2)) & 0x03;
}

void getValuesFromEncodedVector(const EncodedVector* vector, uint8_t** result) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            // Each dimension uses 2 bits
            // Shift and OR the values into the correct position
            result[i][j] = getValueEncodedVector4DQ4Beta2(vector, i, j);
        }
    }
}

void getValuesForEncodedVectorGroup(const EncodedVectorGroup* vector_group, uint8_t*** result) {
    for (int vectorIndex = 0; vectorIndex < 4; vectorIndex++) {
        for (int b_index = 0; b_index < 2; b_index++) {
            for (int d = 0; d < 4; d++) {
                // Each dimension uses 2 bits
                // Shift and OR the values into the correct position
                result[vectorIndex][b_index][d] = (vector_group->b_vectors[vectorIndex * 2 + b_index] >> (d * 2)) & 0x03;
            }
        }
    }
}

void getBetaIndicesForEncodedVectorGroup(const EncodedVectorGroup* vector_group, uint8_t* betaIndices) {
    betaIndices[0] = vector_group->betaIndices & 0x03;
    betaIndices[1] = vector_group->betaIndices >> 2 & 0x03;
    betaIndices[2] = vector_group->betaIndices >> 4 & 0x03;
    betaIndices[3] = vector_group->betaIndices >> 6 & 0x03;
}


uint16_t get_D4_beta_for_index(const uint8_t betaIndex) {
  if (betaIndex < 4) {
      return BETA_VALUES[betaIndex];
  }
  return 0;
}

void print_vector_group(const char* name, const EncodedVectorGroup* vec_group) {
    if (!vec_group) {
        printf("%s = NULL\n", name);
        return;
    }
    
    printf("%s = {\n", name);
    
    // Extract beta indices
    uint8_t beta_indices[4];
    getBetaIndicesForEncodedVectorGroup(vec_group, beta_indices);
    
    // Print each of the 4 subvectors
    for (int subvec = 0; subvec < 4; subvec++) {
        printf("  Subvector %d:\n", subvec);
        
        // Print b₀ for this subvector
        printf("    b₀ = [");
        for (int d = 0; d < 4; d++) {
            uint8_t value = (vec_group->b_vectors[subvec * 2] >> (d * 2)) & 0x03;
            printf("%u", value);
            if (d < 3) {
                printf(", ");
            }
        }
        printf("]\n");
        
        // Print b₁ for this subvector
        printf("    b₁ = [");
        for (int d = 0; d < 4; d++) {
            uint8_t value = (vec_group->b_vectors[subvec * 2 + 1] >> (d * 2)) & 0x03;
            printf("%u", value);
            if (d < 3) {
                printf(", ");
            }
        }
        printf("]\n");
        
        // Print beta value for this subvector
        uint8_t beta_index = beta_indices[subvec];
        uint16_t beta = get_D4_beta_for_index(beta_index);
        printf("    β = %u (%.4f) [index: %u]\n", beta, (float)beta / 500.0f, beta_index);
    }
    
    printf("}\n");
}
