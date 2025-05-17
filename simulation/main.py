import numpy as np
import argparse

import encoder
import simulation
import adaptive_encoding
import adaptive_simulation

def main():
    parser = argparse.ArgumentParser(description='Nested Lattice LUT Encoding/Decoding and Simulation')
    
    # Simulation mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--simulate', action='store_true', help='Run simulation to find optimal beta values')
    mode_group.add_argument('--adaptive', action='store_true', help='Run adaptive beta selection simulation')
    mode_group.add_argument('--optimize-betas', action='store_true', help='Find optimal set of beta values for adaptive encoding')
    
    # Common parameters
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples for simulation')
    parser.add_argument('--min-beta', type=float, default=0.05, help='Minimum beta value to test')
    parser.add_argument('--max-beta', type=float, default=2, help='Maximum beta value to test')
    parser.add_argument('--step-beta', type=float, default=0.07, help='Step size for beta values (for --simulate)')
    parser.add_argument('--std-dev', type=float, default=1.0, help='Standard deviation for normal distribution')
    parser.add_argument('--q', type=int, default=4 * 2 ** 0, help='Quantization parameter')
    parser.add_argument('--M', type=int, default=2, help='Number of encoding steps')
    
    # Adaptive-specific parameters
    parser.add_argument('--num-betas', type=int, default=4, help='Number of beta values for adaptive encoding')
    parser.add_argument('--betas', type=float, nargs='+', help='Specific beta values to use for adaptive encoding')
    parser.add_argument('--no-dither', action='store_true', help='Disable dithering for adaptive encoding')
    parser.add_argument('--no-hadamard', type=bool, default=True, help='Disable Hadamard transform before encoding and after decoding')
    parser.add_argument('--find_best_beta', type=bool, default=False, help='Find the best beta value for the input vector')
    
    args = parser.parse_args()
    
    if args.simulate:
        print("Running simulation to find optimal beta values...")
        optimal_betas = simulation.find_optimal_beta(
            beta_range=(args.min_beta, args.max_beta, args.step_beta),
            num_samples=args.samples,
            dimension=4,
            q=args.q,
            M=args.M,
            std_dev=args.std_dev
        )
        
        print("\nOptimal beta values:")
        print("1. Minimizing MSE:", optimal_betas[0])
        print("2. Minimizing overload errors:", optimal_betas[1])
        print("3. Balancing MSE and overload errors:", optimal_betas[2])
        print("4. Intermediate value with low overload rate:", optimal_betas[3])
    
    elif args.optimize_betas:
        print("Finding optimal set of beta values for adaptive encoding...")
        optimal_betas = adaptive_simulation.optimize_beta_values(
            num_samples=args.samples,
            dimension=4,
            q=args.q,
            M=args.M,
            std_dev=args.std_dev,
            min_beta=args.min_beta,
            max_beta=args.max_beta,
            num_betas=args.num_betas,
            use_dither=not args.no_dither,
            use_hadamard=not args.no_hadamard
        )
        
        print(f"Optimal beta values: {optimal_betas}")
        
        print("\nRunning adaptive simulation with the optimal beta values...")
        results = adaptive_simulation.run_adaptive_simulation(
            num_samples=args.samples,
            dimension=4,
            q=args.q,
            M=args.M,
            beta_values=optimal_betas,
            std_dev=args.std_dev,
            find_best_beta=args.find_best_beta,
            use_dither=not args.no_dither,
            use_hadamard=not args.no_hadamard
        )
        
        adaptive_simulation.plot_adaptive_results(results, optimal_betas)
    
    elif args.adaptive:
        print("Running adaptive beta selection simulation...")
        
        beta_values = args.betas if args.betas else [i for i in [0.113, 0.151, 0.198, 0.36]]
        beta_values = sorted(beta_values)
        
        results = adaptive_simulation.run_adaptive_simulation(
            num_samples=args.samples,
            dimension=4,
            q=args.q,
            M=args.M,
            beta_values=beta_values,
            std_dev=args.std_dev,
            find_best_beta=args.find_best_beta,
            use_dither=not args.no_dither,
            use_hadamard=not args.no_hadamard
        )
        
        adaptive_simulation.plot_adaptive_results(results, beta_values)
    
    # else:
    #     G = np.array([
    #         [-1.0, -1.0, 0.0, 0.0],
    #         [1.0, -1.0, 0.0, 0.0],
    #         [0.0, 1.0, -1.0, 0.0],
    #         [0.0, 0.0, 1.0, -1.0]
    #     ], dtype=np.float32).T
    #     lattice = encoder.Lattice(G)
    #     x = np.array([3.9, 0.0, 1.25, 2.2])
    #     q = 4
    #     M = 2
    #


def multiply_with_generating_matrix(vector):
    generating_matrix = np.array([
        [-1.0, -1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, -1.0]
    ], dtype=np.float32).T
    print(generating_matrix)
    g_inverted = np.linalg.inv(generating_matrix)
    print(g_inverted)

    print(np.dot(g_inverted, np.array([5,4,9,2])))
    vector = np.array(vector, dtype=np.float32)

    if vector.shape != (4,):
        raise ValueError("Input vector must be a 4-element array.")


if __name__ == "__main__":
    main()