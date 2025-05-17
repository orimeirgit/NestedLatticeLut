#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "lut.h"
#include "encoder.h"

#define BATCH_SIZE 10
#define BENCHMARK_MODE 1  // Set to 1 to reduce debug output
#define USE_HADAMARD 1    // Set to 1 to enable Hadamard transform

/**
 * Read vector pairs from file
 */
int read_vector_pairs(const char* filename, float*** vectors_a, float*** vectors_b, int* num_pairs, int* vector_length) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s for reading\n", filename);
        return 0;
    }
    
    // Read header: number of pairs and vector length
    fread(num_pairs, sizeof(int), 1, file);
    fread(vector_length, sizeof(int), 1, file);
    
    printf("File header: %d pairs, vector length %d\n", *num_pairs, *vector_length);
    
    // Allocate memory for vectors
    *vectors_a = (float**)malloc(*num_pairs * sizeof(float*));
    *vectors_b = (float**)malloc(*num_pairs * sizeof(float*));
    
    if (!*vectors_a || !*vectors_b) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return 0;
    }
    
    // Allocate memory for each vector and read from file
    for (int i = 0; i < *num_pairs; i++) {
        (*vectors_a)[i] = (float*)malloc(*vector_length * sizeof(float));
        (*vectors_b)[i] = (float*)malloc(*vector_length * sizeof(float));
        
        if (!(*vectors_a)[i] || !(*vectors_b)[i]) {
            printf("Error: Memory allocation failed\n");
            fclose(file);
            return 0;
        }
        
        fread((*vectors_a)[i], sizeof(float), *vector_length, file);
        fread((*vectors_b)[i], sizeof(float), *vector_length, file);
    }
    
    fclose(file);
    return 1;
}

/**
 * Encode all vectors for LUT-based computation
 */
int encode_vectors_batch(float** vectors, int num_vectors, int vector_length,
                      EncodedVectorGroup*** encoded_vectors, float** vector_norms,
                      int beta_factor, bool use_hadamard) {
    *encoded_vectors = (EncodedVectorGroup**)malloc(num_vectors * sizeof(EncodedVectorGroup*));
    
    if (!*encoded_vectors) {
        printf("Error: Memory allocation failed\n");
        return 0;
    }
    
    // Encode each vector
    for (int i = 0; i < num_vectors; i++) {
        if (!encode_float_vector(vectors[i], vector_length, &(*encoded_vectors)[i], &(*vector_norms)[i])) {
            printf("Failed to encode vector %d\n", i);
            return 0;
        }
        
        // Print progress periodically
        if ((i % 1000 == 0 || i == num_vectors - 1) && num_vectors > 1000) {
            printf("Encoded %d/%d vectors\n", i + 1, num_vectors);
        }
    }
    
    return 1;
}

/**
 * Run exact inner product for a vector
 */
float run_exact_inner_product_for_vector(const float** vectors_a, const float** vectors_b, int idx, int vector_length) {
    float result = 0.0f;
    const float* vector_a = vectors_a[idx];
    const float* vector_b = vectors_b[idx];
    for (int i = 0; i < vector_length; i++) {
        result += vector_a[i] * vector_b[i];
    }
    return result;
}

/**
 * Run benchmark and save results to file
 */
void run_benchmark(const char* input_filename, char* output_filename, bool use_hadamard) {
    // Initialize random seed
    int beta_factor = 500;
    
    float** vectors_a;
    float** vectors_b;
    int num_pairs;
    int vector_length;
    
    printf("Reading vector pairs from file...\n");
    if (!read_vector_pairs(input_filename, &vectors_a, &vectors_b, &num_pairs, &vector_length)) {
        printf("Failed to read vector pairs. Exiting.\n");
        return;
    }
    printf("Read %d vector pairs of length %d successfully.\n", num_pairs, vector_length);

    // Initialize LUT

    printf("Initializing inner product LUT...\n");
    int8_t* lut = init_inner_product_lut();
    if (!lut) {
        printf("Failed to initialize LUT. Exiting.\n");
        return;
    }
    printf("LUT initialized successfully.\n");
    
    // Log Hadamard settings
    printf("Hadamard transform is %s\n", use_hadamard ? "ENABLED" : "DISABLED");
    
    // Encode vectors
    EncodedVectorGroup** encoded_vectors_a;
    EncodedVectorGroup** encoded_vectors_b;
    
    // Allocate memory for vector norms
    float* vector_norms_a = (float*)malloc(num_pairs * sizeof(float));
    float* vector_norms_b = (float*)malloc(num_pairs * sizeof(float));
    
    if (!vector_norms_a || !vector_norms_b) {
        printf("Failed to allocate memory for vector norms. Exiting.\n");
        return;
    }
    
    printf("Encoding vectors...\n");
    if (!encode_vectors_batch(vectors_a, num_pairs, vector_length, &encoded_vectors_a, &vector_norms_a, beta_factor, use_hadamard) ||
        !encode_vectors_batch(vectors_b, num_pairs, vector_length, &encoded_vectors_b, &vector_norms_b, beta_factor, use_hadamard)) {
        printf("Failed to encode vectors. Exiting.\n");
        free(vector_norms_a);
        free(vector_norms_b);
        return;
    }
    printf("Vectors encoded successfully.\n");
    
    // Print an example of a vector group for demonstration
    if (num_pairs > 0 && encoded_vectors_a != NULL && encoded_vectors_a[0] != NULL) {
        printf("\nExample of encoded vector group (first vector, first chunk):\n");
        print_vector_group("Vector Group A[0][0]", &encoded_vectors_a[0][0]);
        printf("Vector A[0] norm: %f\n", vector_norms_a[0]);
    }
    
    if (num_pairs > 0 && encoded_vectors_b != NULL && encoded_vectors_b[0] != NULL) {
        printf("\nExample of encoded vector group (first vector, first chunk):\n");
        print_vector_group("Vector Group B[0][0]", &encoded_vectors_b[0][0]);
        printf("Vector B[0] norm: %f\n", vector_norms_b[0]);
    }
    
    // Open output file
    FILE* output_file = fopen(output_filename, "w");
    if (!output_file) {
        printf("Error: Could not open output file %s\n", output_filename);
        free(vector_norms_a);
        free(vector_norms_b);
        return;
    }
    
    // Write header to output file
    fprintf(output_file, "Batch,Exact_Time_ms,LUT_Time_ms,Speedup,Avg_Error\n");
    
    // Run benchmark in batches
    printf("Running benchmark in batches of %d...\n", BATCH_SIZE);
    
    float beta_factor_2 = (float)beta_factor * (float)beta_factor;
    
    uint16_t beta_mult_values[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            beta_mult_values[i << 2 | j] = (uint16_t)(get_D4_beta_for_index(i) * get_D4_beta_for_index(j));
        }
    }
    
    for (int batch = 0; batch < num_pairs / BATCH_SIZE; batch++) {
        float exact_results[BATCH_SIZE];
        float lut_results[BATCH_SIZE];
        
        const clock_t exact_start = clock();

        for (int vector_idx = 0; vector_idx < BATCH_SIZE; vector_idx++) {
            exact_results[vector_idx] = run_exact_inner_product_for_vector(
                (const float**)vectors_a, (const float**)vectors_b, BATCH_SIZE * batch + vector_idx, vector_length);
        }

        clock_t exact_end = clock();
        const double exact_time = (double) (exact_end - exact_start);
        
        const clock_t lut_start = clock();

        for (int vector_idx = 0; vector_idx < BATCH_SIZE; vector_idx++) {
            int idx = BATCH_SIZE * batch + vector_idx;
            lut_results[vector_idx] = lut_inner_product_for_vector_of_groups(
                (const EncodedVectorGroup**)encoded_vectors_a,
                (const EncodedVectorGroup**)encoded_vectors_b,
                lut, idx, vector_length, beta_mult_values, beta_factor_2,
                vector_norms_a[idx], vector_norms_b[idx]);
        }

        clock_t lut_end = clock();
        const double lut_time = (double) (lut_end - lut_start);
        
        // Calculate speedup and average error
        const double speedup = exact_time / lut_time;
        double total_error = 0.0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            total_error += pow(exact_results[i] - lut_results[i], 2) / (double)vector_length;
        }


        double avg_error = total_error  / (double) BATCH_SIZE;

        // Write batch results to output file
        fprintf(output_file, "%d,%.4f,%.4f,%.2f,%.6f\n",
                batch + 1, (double)(exact_time * 1000) / CLOCKS_PER_SEC, 
                (double)(lut_time * 1000) / CLOCKS_PER_SEC, speedup, avg_error);
        
        // Print progress
        printf("Completed batch %d/%d - Speedup: %.2fx, avg square error: %.6f\n",
               batch + 1, num_pairs / BATCH_SIZE, speedup, avg_error);
    }
    
    fclose(output_file);
    printf("Benchmark completed. Results saved to %s\n", output_filename);
    
    // Free resources
    for (int i = 0; i < num_pairs; i++) {
        free(encoded_vectors_a[i]);
        free(encoded_vectors_b[i]);
        free(vectors_a[i]);
        free(vectors_b[i]);
    }
    free(encoded_vectors_a);
    free(encoded_vectors_b);
    free(vectors_a);
    free(vectors_b);
    free(vector_norms_a);
    free(vector_norms_b);
    free(lut);
}

int main(int argc, char** argv) {
    char* input_filename = "vector_pairs.bin";
    char* output_filename = "benchmark_results.csv";
    bool use_hadamard = USE_HADAMARD;

    // Parse command line arguments
    if (argc > 1) {
        // Display usage information if --help is specified
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            printf("Usage: %s [input_file] [output_file] [hadamard_flag]\n", argv[0]);
            printf("  input_file    : Input binary file containing vector pairs (default: vector_pairs.bin)\n");
            printf("  output_file   : Output CSV file for benchmark results (default: benchmark_results.csv)\n");
            printf("  hadamard_flag : Use Hadamard transform (0=disabled, 1=enabled, default: %d)\n", USE_HADAMARD);
            printf("\nExample: %s my_vectors.bin my_results.csv 1\n", argv[0]);
            return 0;
        }
        
        // Get input filename
        input_filename = argv[1];
    }
    
    if (argc > 2) {
        // Get output filename
        output_filename = argv[2];
    }
    
    if (argc > 3) {
        // Get Hadamard transform flag
        use_hadamard = atoi(argv[3]) != 0;
    }
    
    printf("Benchmark settings:\n");
    printf("  Input file: %s\n", input_filename);
    printf("  Output file: %s\n", output_filename);
    printf("  Hadamard transform: %s\n", use_hadamard ? "Enabled" : "Disabled");
    
    run_benchmark(input_filename, output_filename, use_hadamard);
    
    return 0;
}