import numpy as np
import encoder

def is_in_beta_range(lattice, x, beta, q, M, eps):
    """
    Check if a vector x can be encoded without overload error when scaled by beta.
    
    Args:
        lattice: The lattice structure
        x: Input vector
        beta: Scaling factor
        q: Quantization parameter
        M: Number of encoding steps
        
    Returns:
        True if x can be encoded without overload error, False otherwise
    """
    scaled_x = x / beta
    _, overload_error = encoder.encode(lattice, scaled_x, q, M, eps)
    return not overload_error

def find_adaptive_beta(lattice, x, beta_values, q, M, eps):
    """
    Find the smallest beta from the given values that can encode x without overload error.
    
    Args:
        lattice: The lattice structure
        x: Input vector
        beta_values: List of beta values to choose from (sorted in ascending order)
        q: Quantization parameter
        M: Number of encoding steps
        
    Returns:
        (selected_beta, index) - The selected beta value and its index in beta_values
        If no beta value works, returns the largest beta and its index
    """
    for i, beta in enumerate(beta_values):
        if is_in_beta_range(lattice, x, beta, q, M, eps):
            return beta, i
    
    return beta_values[-1], len(beta_values) - 1


def find_best_beta(lattice, x, beta_values, q, M, eps):
    """
    Find the best beta value from the given list that minimizes the quantization error.

    Args:
        lattice: The lattice structure
        x: Input vector
        beta_values: List of beta values to choose from (sorted in ascending order)
        q: Quantization parameter
        M: Number of encoding steps

    Returns:
        (selected_beta, index) - The selected beta value and its index in beta_values
    """
    best_error = float('inf')
    best_beta = beta_values[0]
    best_index = 0

    for i, beta in enumerate(beta_values):
        scaled_x = x / beta
        encoded_vectors, overload = encoder.encode(lattice, scaled_x, q, M, eps)
        decoded_x = encoder.decode(lattice, encoded_vectors, q, eps)
        error = np.linalg.norm(scaled_x - decoded_x)

        if error < best_error:
            best_error = error
            best_beta = beta
            best_index = i

        if i == len(beta_values) - 1 and overload:
            return beta, i

    return best_beta, best_index


def find_largest_in_range_point(lattice, x, beta, q, M, eps, max_iterations=10, epsilon=10):
    """
    Find the largest point in the direction of x that's still within the beta range.
    This is used when x is too large even for the largest beta.
    
    Args:
        lattice: The lattice structure
        x: Original input vector (too large to be encoded)
        beta: The largest beta value
        q: Quantization parameter
        M: Number of encoding steps
        max_iterations: Maximum number of binary search iterations
        epsilon: Convergence threshold
        
    Returns:
        The scaled vector that can be encoded without overload error
    """
    direction = x / np.linalg.norm(x)
    
    low = 0.0
    high = np.linalg.norm(x)
    best_magnitude = 0.0
    
    for _ in range(max_iterations):
        mid = (low + high) / 2
        test_point = mid * direction
        
        if is_in_beta_range(lattice, test_point, beta, q, M, eps):
            best_magnitude = mid
            low = mid
        else:
            high = mid
            
        if high - low < epsilon:
            break
    
    return best_magnitude * direction


def adaptive_encode_with_dither(lattice, x, beta_values, q, M, eps, should_find_best_beta=False, dither=None):
    """
    Encode a vector using adaptive beta selection and optional dithering.
    
    Args:
        lattice: The lattice structure
        x: Input vector
        beta_values: List of beta values to choose from (sorted in ascending order)
        q: Quantization parameter
        M: Number of encoding steps
        dither: Optional dither vector (if None, no dithering is applied)
        
    Returns:
        (encoded_vectors, selected_beta_index, is_clamped, is_overload)
        - encoded_vectors: The encoded representation
        - selected_beta_index: Index of the beta value used
        - is_clamped: True if the vector was clamped to fit within range
        - is_overload: True if there was an overload error
    """

    if should_find_best_beta:
        beta, beta_index = find_best_beta(lattice, x, beta_values, q, M, eps)
    else:
        beta, beta_index = find_adaptive_beta(lattice, x, beta_values, q, M, eps)

    if beta_index == len(beta_values) - 1 and not is_in_beta_range(lattice, x, beta, q, M, eps):
        x = find_largest_in_range_point(lattice, x, beta, q, M, eps)

    scaled_x = x / beta

    if dither is not None:
        dithered_x = scaled_x - dither
    else:
        dithered_x = scaled_x
    
    encoded_vectors, is_overload = encoder.encode(lattice, dithered_x, q, M, eps)
    
    return encoded_vectors, beta_index, is_overload

def adaptive_decode_with_dither(lattice, encoded_vectors, beta_index, beta_values, q, eps, dither=None):
    """
    Decode a vector that was encoded with adaptive beta selection and optional dithering.
    
    Args:
        lattice: The lattice structure
        encoded_vectors: The encoded representation
        beta_index: Index of the beta value used for encoding
        beta_values: List of beta values
        q: Quantization parameter
        dither: Optional dither vector (if None, no dithering is applied)
        
    Returns:
        The decoded vector
    """
    beta = beta_values[beta_index]
    
    decoded_x = encoder.decode(lattice, encoded_vectors, q, eps)

    if dither is not None:
        decoded_x = decoded_x + dither

    decoded_x = beta * decoded_x
    
    return decoded_x

def generate_dither_vector(lattice):
    """
    Generate a random dither vector in the fundamental Voronoi cell V.
    
    Args:
        lattice: The lattice structure
        
    Returns:
        A random dither vector
    """
    # Simple approximation - generate random vector in the hypercube
    # This is not exactly the fundamental Voronoi cell but a simplification
    return np.zeros(lattice.n)
