import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import encoder
import time

def run_simulation(num_samples=1000, dimension=4, q=4, M=2, beta_values=None, std_dev=1.0):
    """
    Run a simulation with vectors drawn from i.i.d. normal distribution.
    
    Args:
        num_samples: Number of random vectors to generate
        dimension: Dimension of the vectors
        q: Quantization parameter
        M: Number of encoding steps
        beta_values: List of 4 beta values to test
        std_dev: Standard deviation of the normal distribution
    
    Returns:
        Dictionary containing simulation results
    """
    if beta_values is None:
        beta_values = [0.25, 0.5, 0.75, 1.0]  # Default beta values
    
    G = np.array([
        [-1.0, -1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, -1.0]
    ], dtype=np.float32).T
    
    lattice = encoder.Lattice(G)
    
    results = {
        'beta_values': beta_values,
        'mse': [],
        'overload_errors': [],
        'encoding_time': [],
        'decoding_time': []
    }
    
    for beta in beta_values:
        print(f"Testing beta = {beta}")
        mse_sum = 0
        overload_errors = 0
        
        for _ in tqdm(range(num_samples)):
            x = np.random.normal(0, std_dev, dimension)
            x_scaled = x / beta
            t_start = time.time()
            encoded_vectors, overload_error = encoder.encode(lattice, x_scaled, q, M)
            encoding_time = time.time() - t_start

            if overload_error:
                overload_errors += 1

            t_start = time.time()
            decoded_x = encoder.decode(lattice, encoded_vectors, q) * beta
            decoding_time = time.time() - t_start
            
            mse = np.mean((x - decoded_x) ** 2)
            mse_sum += mse
        
        avg_mse = mse_sum / num_samples
        overload_rate = overload_errors / num_samples
        
        results['mse'].append(avg_mse)
        results['overload_errors'].append(overload_rate)
        results['encoding_time'].append(encoding_time)
        results['decoding_time'].append(decoding_time)
        
        print(f"  MSE: {avg_mse:.6f}")
        print(f"  Overload Error Rate: {overload_rate:.4f}")
    
    return results

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot MSE vs beta
    ax1.plot(results['beta_values'], results['mse'], 'o-')
    ax1.set_xlabel('Beta Value')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('MSE vs Beta')
    ax1.grid(True)
    
    # Plot Overload Error Rate vs beta
    ax2.plot(results['beta_values'], results['overload_errors'], 'o-')
    ax2.set_xlabel('Beta Value')
    ax2.set_ylabel('Overload Error Rate')
    ax2.set_title('Overload Error Rate vs Beta')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.show()

def find_optimal_beta(beta_range=None, num_samples=1000, dimension=4, q=4, M=2, std_dev=1.0):
    """
    Find the optimal beta value that minimizes MSE while keeping overload errors below a threshold.
    
    Args:
        beta_range: Range of beta values to test (min, max, step)
        num_samples: Number of random vectors to generate
        dimension: Dimension of the vectors
        q: Quantization parameter
        M: Number of encoding steps
        std_dev: Standard deviation of the normal distribution
    
    Returns:
        Four optimal beta values
    """
    if beta_range is None:
        beta_range = (0.1, 2.0, 0.1)
    
    min_beta, max_beta, step_beta = beta_range
    beta_values = np.arange(min_beta, max_beta + step_beta/2, step_beta)
    
    sim_results = run_simulation(
        num_samples=num_samples, 
        dimension=dimension, 
        q=q, 
        M=M, 
        beta_values=beta_values,
        std_dev=std_dev
    )
    
    plot_results(sim_results)
    
    best_betas = []
    
    # 1. Beta that minimizes MSE
    best_mse_idx = np.argmin(sim_results['mse'])
    best_betas.append(sim_results['beta_values'][best_mse_idx])
    
    # 2. Beta that minimizes overload errors
    best_overload_idx = np.argmin(sim_results['overload_errors'])
    best_betas.append(sim_results['beta_values'][best_overload_idx])
    
    # 3. Beta that balances MSE and overload errors
    # Normalize metrics to [0,1] range
    norm_mse = np.array(sim_results['mse']) / max(sim_results['mse'])
    norm_overload = np.array(sim_results['overload_errors']) / max(sim_results['overload_errors'])
    combined_metric = norm_mse + norm_overload
    best_combined_idx = np.argmin(combined_metric)
    best_betas.append(sim_results['beta_values'][best_combined_idx])
    
    # 4. Intermediate beta value (with low overload error rate)
    # Find betas with overload error rate < 0.1 (or min if all are higher)
    acceptable_betas = [
        beta for beta, err in zip(sim_results['beta_values'], sim_results['overload_errors']) 
        if err < 0.1
    ]
    if acceptable_betas:
        best_betas.append(np.median(acceptable_betas))
    else:
        best_betas.append(sim_results['beta_values'][best_overload_idx])
    
    return best_betas

if __name__ == "__main__":
    # Run a test simulation with default parameters
    optimal_betas = find_optimal_beta(
        beta_range=(0.1, 2.0, 0.1),
        num_samples=500,  # Reduced for faster testing
        dimension=4,
        q=4,
        M=2,
        std_dev=1.0
    )
    
    print("\nOptimal beta values:")
    print("1. Minimizing MSE:", optimal_betas[0])
    print("2. Minimizing overload errors:", optimal_betas[1])
    print("3. Balancing MSE and overload errors:", optimal_betas[2])
    print("4. Intermediate value with low overload rate:", optimal_betas[3]) 