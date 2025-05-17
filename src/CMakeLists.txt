cmake_minimum_required(VERSION 3.10)
project(NestedLatticLUT C)

set(CMAKE_C_STANDARD 11)

# Set optimization level for release builds
set(CMAKE_C_FLAGS_RELEASE "-O3")

# Common source files
set(COMMON_SOURCES
    encoder.c
    decoder.c
    utility.c
    encoded_vectors.c
)

# Add benchmark executable
add_executable(BenchmarkLUT
    benchmark.c
    ${COMMON_SOURCES}
    lut.c
)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Link math library to all executables
target_link_libraries(BenchmarkLUT m)
