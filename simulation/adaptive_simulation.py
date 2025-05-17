import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import encoder
import adaptive_encoding
import random


def generate_hadamard_matrix(n):
    """
    Generate a Hadamard matrix of size n x n.
    n must be a power of 2.
    
    Args:
        n: Size of the matrix (must be a power of 2)
        
    Returns:
        Hadamard matrix of size n x n
    """
    if n == 1:
        return np.array([[1]])
    
    # Check if n is a power of 2
    if n & (n - 1) != 0:
        raise ValueError("Size must be a power of 2")
    
    # Recursive construction using Sylvester's method
    h = generate_hadamard_matrix(n // 2)
    return np.block([[h, h], [h, -h]])


def run_adaptive_simulation(num_samples=1000, dimension=4, q=4, M=2, beta_values=None, std_dev=1.0, find_best_beta=False, use_dither=True, use_hadamard=True):
    """
    Run a simulation with vectors drawn from i.i.d. normal distribution,
    using adaptive beta selection.
    
    Args:
        num_samples: Number of random vectors to generate
        dimension: Dimension of the vectors
        q: Quantization parameter
        M: Number of encoding steps
        beta_values: List of beta values to choose from (sorted in ascending order)
        std_dev: Standard deviation of the normal distribution
        use_dither: Whether to use dithering
        use_hadamard: Whether to use Hadamard transform before encoding and after decoding
        
    Returns:
        Dictionary containing simulation results
    """
    if beta_values is None:
        beta_values = [0.25, 0.5, 0.75, 1.0]  # Default beta values

    beta_factor = 500

    beta_values = [i * beta_factor for i in beta_values]

    G = np.array([
        [-1.0, -1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, -1.0]
    ], dtype=np.float32).T

    max_mean = 0

    lattice = encoder.Lattice(G)
    
    # Generate Hadamard matrix if needed
    if use_hadamard:
        # Ensure dimension is a power of 2
        hadamard_size = 1
        while hadamard_size < dimension:
            hadamard_size *= 2
            
        H = generate_hadamard_matrix(hadamard_size)
        # If dimension is not a power of 2, use the submatrix of size dimension x dimension
        if hadamard_size > dimension:
            H = H[:dimension, :dimension]
            
        # Normalize the Hadamard matrix
        H_normalized = H / np.sqrt(dimension)
        
        # For Hadamard matrices, H^(-1) = H^T / n = H / n (since H is symmetric)
        H_inv = H_normalized

    results = {
        'beta_usage': np.zeros(len(beta_values)),
        'total_mse': 0,
        'beta_mse': [[] for _ in range(len(beta_values))],
        'overload_errors': 0,
        'clamped_vectors': 0,
        'encoding_time': 0,
        'decoding_time': 0,
        'beta_distribution': [[] for _ in range(len(beta_values))],
        'vector_norms': []
    }

    print(f"Running adaptive simulation with {num_samples} samples")
    print(f"Beta values: {beta_values}")
    print(f"Using dithering: {use_dither}")
    print(f"Using Hadamard transform: {use_hadamard}")

    eps = [-0.345e-8, -0.867e-8, -0.567e-8, 0.939e-8]
    for i in tqdm(range(num_samples)):
        x = np.random.normal(0, std_dev, dimension) * beta_factor
        vector_norm = np.linalg.norm(x / 500)
        results['vector_norms'].append(vector_norm)

        # Apply Hadamard transform before encoding if enabled
        if use_hadamard:
            x_hadamard = np.dot(H_normalized, x)
        else:
            x_hadamard = x

        if use_dither:
            dither = adaptive_encoding.generate_dither_vector(lattice)
        else:
            dither = None

        t_start = time.time()
        encoded_vectors, beta_index, is_overload = adaptive_encoding.adaptive_encode_with_dither(
            lattice, x_hadamard, beta_values, q, M, eps, find_best_beta, dither)
        encoding_time = time.time() - t_start

        results['beta_usage'][beta_index] += 1
        results['beta_distribution'][beta_index].append(vector_norm)

        if is_overload:
            results['overload_errors'] += 1

        t_start = time.time()
        decoded_x_hadamard = adaptive_encoding.adaptive_decode_with_dither(
            lattice, encoded_vectors, beta_index, beta_values, q, eps, dither)
        
        # Apply inverse Hadamard transform after decoding if enabled
        if use_hadamard:
            decoded_x = np.dot(H_inv, decoded_x_hadamard)
        else:
            decoded_x = decoded_x_hadamard

        if ((decoded_x / beta_factor) ** 2).sum() > max_mean:
            print(decoded_x / beta_factor)
            max_mean = ((decoded_x / beta_factor) ** 2).sum()
            print(max_mean)

        decoding_time = time.time() - t_start

        mse = np.mean((x / beta_factor - decoded_x / beta_factor) ** 2)
        results['total_mse'] += mse
        results['beta_mse'][beta_index].append(mse)

        results['encoding_time'] += encoding_time
        results['decoding_time'] += decoding_time

    results['avg_mse'] = results['total_mse'] / num_samples
    results['overload_rate'] = results['overload_errors'] / num_samples
    results['avg_encoding_time'] = results['encoding_time'] / num_samples
    results['avg_decoding_time'] = results['decoding_time'] / num_samples
    results['beta_usage_percentage'] = results['beta_usage'] / num_samples * 100
    results['avg_beta_mse'] = [np.mean(mse_list) if len(mse_list) > 0 else 0 for mse_list in results['beta_mse']]

    print("\nSimulation Results:")
    print(f"Average MSE: {results['avg_mse']:.6f}")
    print(f"Overload Error Rate: {results['overload_rate']:.4f}")
    print(f"Beta Usage (%):")
    for i, beta in enumerate(beta_values):
        print(
            f"  Beta {beta:.2f}: {results['beta_usage_percentage'][i]:.2f}% (Avg MSE: {results['avg_beta_mse'][i]:.6f})")

    return results


def plot_adaptive_results(results, beta_values):
    """Plot the results of the adaptive simulation."""
    # Create figure with 2 rows, 2 columns
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Beta usage distribution
    ax1.bar(range(len(beta_values)), results['beta_usage_percentage'])
    ax1.set_xticks(range(len(beta_values)))
    ax1.set_xticklabels([f"{beta:.2f}" for beta in beta_values])
    ax1.set_xlabel('Beta Value')
    ax1.set_ylabel('Usage Percentage (%)')
    ax1.set_title('Beta Usage Distribution')
    ax1.grid(True, axis='y')

    # Plot 2: Vector norm histograms by selected beta
    colors = ['blue', 'green', 'orange', 'red']
    for i, beta in enumerate(beta_values):
        if len(results['beta_distribution'][i]) > 0:
            ax2.hist(results['beta_distribution'][i], bins=20, alpha=0.5,
                     label=f'Beta {beta:.2f}', color=colors[i % len(colors)])
    ax2.set_xlabel('Vector Norm')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Vector Norm Distribution by Selected Beta')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Vector norm histogram (overall)
    ax3.hist(results['vector_norms'], bins=30)
    ax3.set_xlabel('Vector Norm')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Overall Vector Norm Distribution')
    ax3.grid(True)

    # Plot 4: Pie chart of beta usage
    labels = [f"Beta {beta:.2f}" for beta in beta_values]
    ax4.pie(results['beta_usage'], labels=labels, autopct='%1.1f%%',
            colors=colors[:len(beta_values)])
    ax4.set_title('Beta Usage Proportion')

    plt.tight_layout()
    plt.savefig('adaptive_simulation_results.png')
    plt.show()

    # Plot 5: Mse plot.
    plt.figure(figsize=(10, 6))
    mse_data = [mse_list for mse_list in results['beta_mse'] if len(mse_list) > 0]
    beta_labels = [f"Beta {beta:.2f}" for i, beta in enumerate(beta_values) if len(results['beta_mse'][i]) > 0]

    bp = plt.boxplot(mse_data, patch_artist=True)

    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=colors[i % len(colors)], alpha=0.7)

    avg_mse_values = [np.mean(mse_list) for mse_list in mse_data]
    plt.plot(range(1, len(avg_mse_values) + 1), avg_mse_values, 'ro-', label='Average MSE')

    plt.xticks(range(1, len(beta_labels) + 1), beta_labels)
    plt.xlabel('Beta Value')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE Distribution by Beta Value')
    plt.grid(True, axis='y')
    plt.legend()

    plt.tight_layout()
    plt.savefig('beta_mse_distribution.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(beta_values)), results['avg_beta_mse'], color=colors[:len(beta_values)])
    plt.xticks(range(len(beta_values)), [f"Beta {beta:.2f}" for beta in beta_values])
    plt.xlabel('Beta Value')
    plt.ylabel('Average MSE')
    plt.title('Average MSE by Beta Value')
    plt.grid(True, axis='y')

    for i, v in enumerate(results['avg_beta_mse']):
        if v > 0:
            plt.text(i, v + 0.0001, f"{v:.6f}", ha='center', va='bottom', rotation=45, fontsize=9)

    plt.tight_layout()
    plt.savefig('beta_avg_mse.png')
    plt.show()


def optimize_beta_values(num_samples=1000, dimension=4, q=4, M=2, std_dev=1.0,
                         min_beta=0.1, max_beta=2.0, num_betas=4, use_dither=True, use_hadamard=True):
    """
    Find optimal set of beta values for adaptive encoding.
    
    Args:
        num_samples: Number of random vectors to generate
        dimension: Dimension of the vectors
        q: Quantization parameter
        M: Number of encoding steps
        std_dev: Standard deviation of the normal distribution
        min_beta: Minimum beta value
        max_beta: Maximum beta value
        num_betas: Number of beta values to find
        use_dither: Whether to use dithering
        use_hadamard: Whether to use Hadamard transform
        
    Returns:
        List of optimized beta values
    """
    G = np.array([
        [-1.0, -1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, -1.0]
    ], dtype=np.float32).T
    lattice = encoder.Lattice(G)
    
    # Generate Hadamard matrix if needed
    if use_hadamard:
        # Ensure dimension is a power of 2
        hadamard_size = 1
        while hadamard_size < dimension:
            hadamard_size *= 2
            
        H = generate_hadamard_matrix(hadamard_size)
        # If dimension is not a power of 2, use the submatrix of size dimension x dimension
        if hadamard_size > dimension:
            H = H[:dimension, :dimension]
            
        # Normalize the Hadamard matrix
        H_normalized = H / np.sqrt(dimension)
    else:
        H_normalized = None

    vectors = np.random.normal(0, std_dev, (num_samples, dimension))
    
    # Apply Hadamard transform if enabled
    if use_hadamard:
        transformed_vectors = np.array([np.dot(H_normalized, v) for v in vectors])
    else:
        transformed_vectors = vectors
    
    vector_norms = np.linalg.norm(transformed_vectors, axis=1)
    sorted_indices = np.argsort(vector_norms)
    sorted_vectors = transformed_vectors[sorted_indices]

    cutoffs = np.linspace(0, num_samples - 1, num_betas + 1).astype(int)[1:]
    beta_vectors = [sorted_vectors[i] for i in cutoffs]

    optimal_betas = []
    for vec in beta_vectors:
        left = min_beta
        right = max_beta
        best_beta = max_beta
        epsilon = 1e-2

        while right - left > epsilon:
            mid = (left + right) / 2
            if adaptive_encoding.is_in_beta_range(lattice, vec, mid, q, M, eps=1e-10 * np.random.normal(0, 1, size=lattice.n)):
                best_beta = mid
                right = mid
            else:
                left = mid

        optimal_betas.append(best_beta * 1.05)

    optimal_betas = np.clip(sorted(optimal_betas), min_beta, max_beta)

    return optimal_betas.tolist()


if __name__ == "__main__":
    print("Finding optimal beta values...")
    optimal_betas = optimize_beta_values(
        num_samples=2000,
        dimension=4,
        q=4,
        M=2,
        std_dev=1.0,
        min_beta=0.1,
        max_beta=2.0,
        num_betas=4,
        use_dither=True,
        use_hadamard=True
    )

    print(f"Optimal beta values: {optimal_betas}")

    results = run_adaptive_simulation(
        num_samples=1000,
        dimension=4,
        q=4,
        M=2,
        beta_values=optimal_betas,
        std_dev=1.0,
        use_dither=True,
        use_hadamard=True
    )

    plot_adaptive_results(results, optimal_betas)
