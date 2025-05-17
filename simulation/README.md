# Nested Lattice Simulation

This project implements a simulation framework to find optimal beta scaling values.

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Encoding/Decoding Example

Run the basic example:

```bash
python main.py
```

This will encode and decode a predefined vector using the lattice structure and show the results. It will also demonstrate the new adaptive beta selection mechanism on the same vector.

### Standard Simulation Mode

To run the standard simulation to find optimal beta values:

```bash
python main.py --simulate
```

#### Simulation Parameters

You can customize the simulation with the following parameters:

- `--samples`: Number of random vectors to generate (default: 500)
- `--min-beta`: Minimum beta value to test (default: 0.1)
- `--max-beta`: Maximum beta value to test (default: 2.0)
- `--step-beta`: Step size for beta values (default: 0.1)
- `--std-dev`: Standard deviation for normal distribution (default: 1.0)
- `--q`: Quantization parameter (default: 4)
- `--M`: Number of encoding steps (default: 2)

Example with custom parameters:

```bash
python main.py --simulate --samples 1000 --min-beta 0.2 --max-beta 1.5 --step-beta 0.05 --std-dev 0.5
```

### Adaptive Beta Selection

The project includes an adaptive beta selection mechanism that chooses the smallest beta from a set of predefined values that works for a given input vector. For vectors that exceed even the largest beta's range, it finds the largest point in the lattice that's still within range.

#### Running the Adaptive Simulation

To run the adaptive beta selection simulation:

```bash
python main.py --adaptive
```

You can specify your own beta values:

```bash
python main.py --adaptive --betas 0.3 0.6 0.9 1.2
```

#### Optimizing Beta Values

To find an optimal set of beta values for adaptive encoding:

```bash
python main.py --optimize-betas
```

This will:
1. Generate random vectors from a normal distribution
2. Find the optimal set of beta values based on the distribution of vectors
3. Run a simulation with these optimal values
4. Plot the results

Additional parameters for adaptive simulation:

- `--num-betas`: Number of beta values to find (default: 4)
- `--no-dither`: Disable dithering for adaptive encoding

### Output

The standard simulation will output:

1. A plot showing MSE (Mean Squared Error) and Overload Error Rate vs. Beta values
2. Four recommended beta values optimized for different criteria:
   - Beta that minimizes MSE
   - Beta that minimizes overload errors
   - Beta that balances MSE and overload errors
   - Intermediate beta value with low overload rate

The adaptive simulation will output:

1. A dashboard of 4 plots showing:
   - Beta usage distribution
   - Vector norm distribution by selected beta
   - Overall vector norm distribution
   - Beta usage proportion
2. Statistics on:
   - Average MSE
   - Overload error rate
   - Clamped vector rate
   - Beta usage percentages

## Project Structure

- `encoder.py`: Contains the lattice encoding and decoding algorithms
- `simulation.py`: Implements the standard simulation framework to find optimal beta values
- `adaptive_encoding.py`: Implements the adaptive beta selection and encoding/decoding with dithering
- `adaptive_simulation.py`: Provides simulation framework for testing adaptive beta selection
- `main.py`: Main script with CLI interface to run examples and simulations

## How It Works

The system works by:

1. Creating a lattice structure with a generating matrix
2. Encoding vectors into a sequence of integer vectors
3. Decoding vectors back to reconstruct the original input
4. The beta parameter scales the input vector before encoding, affecting the trade-off between MSE and overload errors

### Adaptive Beta Selection

The adaptive beta selection mechanism:

1. Takes a set of sorted beta values (e.g., [0.25, 0.5, 0.75, 1.0])
2. For each input vector, it finds the smallest beta that can encode the vector without overload error
3. If the vector is too large for even the largest beta, it finds the largest point in the direction of the vector that can be encoded
4. Optional dithering shifts the input by a random vector to improve statistical properties

This approach optimizes the trade-off between distortion and overload probability by using different beta values for different input vectors, based on their magnitude. 
