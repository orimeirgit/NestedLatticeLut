#!/usr/bin/env python3
"""
Generate vector pairs from a normal distribution for benchmark testing.

This script creates a binary file containing pairs of long vectors
drawn from a normal distribution. The format matches what the benchmark.c
program expects.

Usage:
  python generate_vectors.py [output_file] [num_pairs] [vector_length] [mean] [stddev]

Arguments:
  output_file   - Output binary file (default: vector_pairs.bin)
  num_pairs     - Number of vector pairs to generate (default: 1000)
  vector_length - Length of each vector (default: 1024)
  mean          - Mean of the normal distribution (default: 0.0)
  stddev        - Standard deviation of the normal distribution (default: 1.0)
"""

import numpy as np
import struct
import sys
import os

def generate_vector_pairs(output_file, num_pairs, vector_length, mean, stddev):
    """Generate vector pairs from a normal distribution and save to binary file."""
    
    print(f"Generating {num_pairs} vector pairs of length {vector_length} from normal distribution (mean={mean}, stddev={stddev})...")
    
    # Generate random vectors from normal distribution
    # Each vector has vector_length elements
    vectors_a = np.random.normal(mean, stddev, (num_pairs, vector_length)).astype(np.float32)
    vectors_b = np.random.normal(mean, stddev, (num_pairs, vector_length)).astype(np.float32)
    
    # Write vectors to binary file
    with open(output_file, 'wb') as f:
        # Write header: number of pairs and vector length (2 4-byte integers)
        f.write(struct.pack('ii', num_pairs, vector_length))
        
        # Write vector pairs as interleaved vectors (each float is 4 bytes)
        for i in range(num_pairs):
            if i % 10000 == 0:
                print(f"Writing pair {i+1}/{num_pairs}...")
            # Write vector A
            for j in range(vector_length):
                f.write(struct.pack('f', vectors_a[i, j]))
            
            # Write vector B
            for j in range(vector_length):
                f.write(struct.pack('f', vectors_b[i, j]))
    
    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"Generated {num_pairs} vector pairs successfully.")
    print(f"Output file: {output_file} (Size: {file_size_mb:.2f} MB)")
    print(f"Format: 2 int32 header + {num_pairs} pairs of float32[{vector_length}] vectors")
    
    # Print some statistics about the generated data
    print("\nVector statistics:")
    print(f"Vector A - Mean: {np.mean(vectors_a):.6f}, Std: {np.std(vectors_a):.6f}")
    print(f"Vector B - Mean: {np.mean(vectors_b):.6f}, Std: {np.std(vectors_b):.6f}")
    
    # Print a few sample vector elements (first 3 pairs, first 10 elements if vector is long)
    print("\nSample vectors (first 3 pairs, showing first 10 elements if vector is long):")
    for i in range(min(3, num_pairs)):
        print(f"Pair {i+1}:")
        print(f"  A: {vectors_a[i, :min(10, vector_length)]}{'...' if vector_length > 10 else ''}")
        print(f"  B: {vectors_b[i, :min(10, vector_length)]}{'...' if vector_length > 10 else ''}")
    
    print("\nTo run benchmarks with this file, use:")
    print(f"./BenchmarkLUT {output_file} results.csv 1")

if __name__ == "__main__":
    # Parse command line arguments
    output_file = "./vector_pairs.bin"  # Changed default to current directory
    num_pairs = 1000
    vector_length = 1024  # Default to 1024 (2^10)
    mean = 0.0
    stddev = 1.0
    
    # Display help if requested
    if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print(__doc__)
        sys.exit(0)
    
    # Override defaults with command line arguments if provided
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    if len(sys.argv) > 2:
        num_pairs = int(sys.argv[2])
    if len(sys.argv) > 3:
        vector_length = int(sys.argv[3])
    if len(sys.argv) > 4:
        mean = float(sys.argv[4])
    if len(sys.argv) > 5:
        stddev = float(sys.argv[5])
    
    generate_vector_pairs(output_file, num_pairs, vector_length, mean, stddev) 