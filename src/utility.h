#ifndef NESTEDLATTICLUT_UTILITY_H
#define NESTEDLATTICLUT_UTILITY_H

#include <stdint.h>
#include "encoder.h"

/**
 * @brief Global epsilon vector for numerical stability
 * This small constant vector is added to input vectors in nearest_lattice_point
 * to avoid numerical edge cases.
 */
extern float global_epsilon[4];

typedef struct {
    float beta;        // Scaling factor β > 0
    float* dither;     // Dither vector z ∈ V (dimension d)
    bool use_dither;   // Whether to use dithering
} ScalingDithering;


bool init_scaling_dithering(ScalingDithering* sd, float beta, const float* dither, int dimension);

void free_scaling_dithering(ScalingDithering* sd);

void apply_pre_encoding(const float* x, const ScalingDithering* sd, float* result, int dimension);

void apply_post_decoding(const float* x, const ScalingDithering* sd, float* result, int dimension);

void matrix_vector_multiply(const float* mat, const float* x, float* y, int rows, int cols);
void int_matrix_vector_multiply(const float* mat, const float* x, int16_t* y, int rows, int cols);

void vector_mod_q(const int* v, uint8_t* result, const int dimension, int q);

void d4_nearest_lattice_point(const float* x, float* result);

void round_vector(const float* x, float* result, int dimension);
void round_vector_to_int(const float* x, int* result, int dimension);

void vector_multiply(const float* x, float* y, int dimension, float scalar);

void vector_divide(const float* x, float* y, int dimension, float scalar);

void vector_add(const float* x, const float* y, float* z, int dimension);

void float_vector_subtract(const float* x, const float* y, float* z, int dimension);

void vector_normalize(const float* x, float* result, int dimension);

float vector_norm(const float* x, int dimension);

void print_int_vector(const char* name, const int* vec, int dimension);
void print_uint8_vector(const char* name, const uint8_t* vec, int dimension);
void print_uint32_vector(const char* name, const uint32_t* vec, int dimension);
void print_float_vector(const char* name, const float* vec, int dimension);
void print_int16_vector(const char* name, const int16_t* vec, const int dimension);
void print_int8_vector(const char* name, const int8_t* vec, const int dimension);

void apply_hadamard_transform(const float* x, float* result, int dimension);

#endif