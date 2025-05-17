#include "utility.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/**
 * Global epsilon vector for numerical stability
 */
float global_epsilon[4] = {-0.345e-5f, -0.867e-5f, -0.567e-6f,0.939e-6f};

/**
 * Initialize global epsilon with small random values
 */
void init_global_epsilon(float scale) {
    // Seed the random number generator if not already done
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < 4; i++) {
        float random_value = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        global_epsilon[i] = random_value;
    }
}

/**
 * Print the global epsilon vector
 */
void print_global_epsilon(void) {
    printf("Global epsilon = [%.10e, %.10e, %.10e, %.10e]\n", 
           global_epsilon[0], global_epsilon[1], 
           global_epsilon[2], global_epsilon[3]);
}

void matrix_vector_multiply(const float* mat, const float* x, float* y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            y[i] += mat[i * cols + j] * x[j];
        }
    }
}

void int_matrix_vector_multiply(const float* mat, const float* x, int16_t* y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0;
        for (int j = 0; j < cols; j++) {
            y[i] += lroundf(mat[i * cols + j] * x[j]);
        }
    }
}

void vector_mod_q(const int* v, uint8_t* result, const int dimension, int q) {
    for (int i = 0; i < dimension; i++) {
        int value = v[i] % q;

        if (value < 0) {
            value += q;
        }

        result[i] = value;
    }
}

void round_vector(const float* x, float* result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        result[i] = roundf(x[i]);
    }
}

void int16_round_vector(const float* x, int16_t* result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        result[i] = lroundf(x[i]);
    }
}

void round_vector_to_int(const float* x, int* result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        const float n = x[i];
        result[i] = (int)(n < 0 ? (n - 0.5) : (n + 0.5));
    }
}

void abs_vector(const float* x, float* result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        result[i] = fabsf(x[i]);
    }
}

static float round_to_the_second_close_integer(const float n) {
    if (roundf(n) < n) {
        return floorf(n) + 1;
    }
    return floorf(n);
}

static int16_t int16_round_to_the_second_close_integer(const float n) {
    if (roundf(n) < n) {
        return (int16_t) n + 1;
    }
    return (int16_t) n;
}

void d4_nearest_lattice_point(const float* x, float* result) {
    float* rounded_vector = (float*)malloc(D * sizeof(float));
    float* error_vector = (float*)malloc(D * sizeof(float));
    float* abs_error = (float*)malloc(D * sizeof(float));
    float* x_with_epsilon = (float*)malloc(D * sizeof(float));
    
    if (!rounded_vector || !error_vector || !abs_error || !x_with_epsilon) {
        free(rounded_vector);
        free(error_vector);
        free(abs_error);
        free(x_with_epsilon);
        return;
    }

    // Add epsilon to the input vector for numerical stability
    for (int i = 0; i < D; i++) {
        x_with_epsilon[i] = x[i] + global_epsilon[i % 4]; // Use modulo to handle dimensions > 4
    }

    round_vector(x_with_epsilon, rounded_vector, D);
    float_vector_subtract(x_with_epsilon, rounded_vector, error_vector, D);
    abs_vector(error_vector, abs_error, D);

    float maximal_error = 0.0f;
    int maximal_error_index = 0;
    int rounded_sum = 0;

    for (int i = 0; i < D; i++) {
        if (abs_error[i] > maximal_error) {
            maximal_error = abs_error[i];
            maximal_error_index = i;
        }
        rounded_sum += (int)rounded_vector[i];
        result[i] = rounded_vector[i];
    }

    if (rounded_sum % 2 != 0) {
        result[maximal_error_index] = round_to_the_second_close_integer(x_with_epsilon[maximal_error_index]);
    }

    free(rounded_vector);
    free(error_vector);
    free(abs_error);
    free(x_with_epsilon);
}

void int16_nearest_lattice_point(const float* x, int16_t* result) {
    float* rounded_vector = (float*)malloc(D * sizeof(float));
    float* error_vector = (float*)malloc(D * sizeof(float));
    float* abs_error = (float*)malloc(D * sizeof(float));
    float* x_with_epsilon = (float*)malloc(D * sizeof(float));

    if (!rounded_vector || !error_vector || !abs_error || !x_with_epsilon) {
        free(rounded_vector);
        free(error_vector);
        free(abs_error);
        free(x_with_epsilon);
        return;
    }

    // Add epsilon to the input vector for numerical stability
    for (int i = 0; i < D; i++) {
        x_with_epsilon[i] = x[i] + global_epsilon[i % 4]; // Use modulo to handle dimensions > 4
    }

    round_vector(x_with_epsilon, rounded_vector, D);
    float_vector_subtract(x_with_epsilon, rounded_vector, error_vector, D);
    abs_vector(error_vector, abs_error, D);

    float maximal_error = 0.0f;
    int maximal_error_index = 0;
    int rounded_sum = 0;

    for (int i = 0; i < D; i++) {
        if (abs_error[i] > maximal_error) {
            maximal_error = abs_error[i];
            maximal_error_index = i;
        }
        rounded_sum += (int)rounded_vector[i];
        result[i] = (int16_t) rounded_vector[i];
    }

    if (rounded_sum % 2 != 0) {
        result[maximal_error_index] = round_to_the_second_close_integer(x_with_epsilon[maximal_error_index]);
    }

    free(rounded_vector);
    free(error_vector);
    free(abs_error);
    free(x_with_epsilon);
}

void vector_multiply(const float* x, float* y, int dimension, float scalar) {
    for (int i = 0; i < dimension; i++) {
        y[i] = x[i] * scalar;
    }
}

void vector_divide(const float* x, float* y, int dimension, float scalar) {
    for (int i = 0; i < dimension; i++) {
        y[i] = x[i] / scalar;
    }
}

void int16_vector_divide(const int16_t* x, float* y, int dimension, float scalar) {
    for (int i = 0; i < dimension; i++) {
        y[i] = ((float) x[i]) / scalar;
    }
}

void vector_add(const float* x, const float* y, float* z, int dimension) {
    for (int i = 0; i < dimension; i++) {
        z[i] = x[i] + y[i];
    }
}

void float_vector_subtract(const float* x, const float* y, float* z, int dimension) {
    for (int i = 0; i < dimension; i++) {
        z[i] = x[i] - y[i];
    }
}

void int16_vector_subtract(const int16_t* x, const int16_t* y, int16_t* z, int dimension) {
    for (int i = 0; i < dimension; i++) {
        z[i] = x[i] - y[i];
    }
}

float vector_norm(const float* x, int dimension) {
    float sum_squared = 0.0f;
    for (int i = 0; i < dimension; i++) {
        sum_squared += x[i] * x[i];
    }
    return sqrtf(sum_squared);
}

void vector_normalize(const float* x, float* result, int dimension) {
    float norm = vector_norm(x, dimension);
    if (norm < 1e-10f) {
        for (int i = 0; i < dimension; i++) {
            result[i] = 0.0f;
        }
    } else {
        for (int i = 0; i < dimension; i++) {
            result[i] = x[i] / norm;
        }
    }
}

void print_int_vector(const char* name, const int* vec, const int dimension) {
    printf("%s = [", name);
    for (int i = 0; i < dimension; i++) {
        printf("%d", vec[i]);
        if (i < dimension - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void print_int16_vector(const char* name, const int16_t* vec, const int dimension) {
    printf("%s = [", name);
    for (int i = 0; i < dimension; i++) {
        printf("%d", vec[i]);
        if (i < dimension - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}


void print_uint32_vector(const char* name, const uint32_t* vec, const int dimension) {
    printf("%s = [", name);
    for (int i = 0; i < dimension; i++) {
        printf("%u", vec[i]);
        if (i < dimension - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void print_uint8_vector(const char* name, const uint8_t* vec, const int dimension) {
    printf("%s = [", name);
    for (int i = 0; i < dimension; i++) {
        printf("%u", vec[i]);
        if (i < dimension - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void print_int8_vector(const char* name, const int8_t* vec, const int dimension) {
    printf("%s = [", name);
    for (int i = 0; i < dimension; i++) {
        printf("%d", vec[i]);
        if (i < dimension - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}


void print_float_vector(const char* name, const float* vec, const int dimension) {
    printf("%s = [", name);
    for (int i = 0; i < dimension; i++) {
        printf("%.5f", vec[i]);
        if (i < dimension - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

bool init_scaling_dithering(ScalingDithering* sd, const float beta, const float* dither, const int d) {
    if (!sd || beta <= 0.0f) {
        return false;
    }
    
    sd->beta = beta;
    sd->use_dither = (dither != NULL);
    
    if (sd->use_dither) {
        sd->dither = (float*)malloc(d * sizeof(float));
        if (!sd->dither) {
            return false;
        }
        memcpy(sd->dither, dither, d * sizeof(float));
    } else {
        sd->dither = NULL;
    }
    
    return true;
}

void free_scaling_dithering(ScalingDithering* sd) {
    if (!sd) {
        return;
    }
    
    if (sd->dither) {
        free(sd->dither);
        sd->dither = NULL;
    }
}

void apply_pre_encoding(const float* x, const ScalingDithering* sd, float* result, const int d) {
    if (sd->use_dither) {
        float_vector_subtract(x, sd->dither, result, d);
    } else {
        memcpy(result, x, d * sizeof(float));
    }
    
    for (int i = 0; i < d; i++) {
        result[i] /= sd->beta;
    }
}

void apply_post_decoding(const float* x, const ScalingDithering* sd, float* result, const int d) {
    for (int i = 0; i < d; i++) {
        result[i] = x[i] * sd->beta;
    }
    
    if (sd->use_dither) {
        float* temp = (float*)malloc(d * sizeof(float));
        if (!temp) {
            return;
        }
        
        vector_add(result, sd->dither, temp, d);

        memcpy(result, temp, d * sizeof(float));
        
        free(temp);
    }
}

/**
 * Generate a Hadamard matrix of size n x n (n must be a power of 2)
 */
float* generate_hadamard_matrix(int n) {
    // Check if n is a power of 2
    if (n <= 0 || (n & (n - 1)) != 0) {
        printf("Error: Hadamard matrix size must be a power of 2\n");
        return NULL;
    }
    
    float* matrix = (float*)malloc(n * n * sizeof(float));
    if (!matrix) {
        printf("Error: Memory allocation failed for Hadamard matrix\n");
        return NULL;
    }
    
    // Initialize with H(1) = [1]
    matrix[0] = 1.0f;
    
    // Build Hadamard matrix using recursive doubling
    for (int k = 1; k < n; k *= 2) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                // Copy the quadrants using the Sylvester construction
                matrix[(i+k)*n + j]    = matrix[i*n + j];    // Top-right quadrant
                matrix[i*n + (j+k)]    = matrix[i*n + j];    // Bottom-left quadrant
                matrix[(i+k)*n + (j+k)] = -matrix[i*n + j];  // Bottom-right quadrant (negated)
            }
        }
    }
    
    // Normalize the matrix by 1/sqrt(n) to make it orthogonal
    float norm_factor = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n*n; i++) {
        matrix[i] *= norm_factor;
    }
    
    return matrix;
}

void free_hadamard_matrix(float* matrix) {
    if (matrix) {
        free(matrix);
    }
}

/**
 * Apply Hadamard transform to a vector
 */
void apply_hadamard_transform(const float* x, float* result, int dimension) {
    // Check if dimension is a power of 2
    if (dimension <= 0 || (dimension & (dimension - 1)) != 0) {
        printf("Error: Vector dimension must be a power of 2 for Hadamard transform\n");
        return;
    }
    
    float* hadamard = generate_hadamard_matrix(dimension);
    if (!hadamard) {
        return;
    }
    
    // Multiply: result = H * x
    matrix_vector_multiply(hadamard, x, result, dimension, dimension);
    
    free_hadamard_matrix(hadamard);
}