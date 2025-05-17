import numpy as np

class Lattice:

    def __init__(self, G):
        self.G = G
        self.n = G.shape[0]
        self.invG = np.linalg.inv(G)

    def nearest_lattice_point(self, x):
        rounded = np.round(x).astype(int)
        if np.isclose(rounded.sum() % 2, 0, atol=1e-5):
            return rounded
        error = x - rounded
        max_error = np.argmax(np.abs(error))
        rounded[max_error] += 1 if x[max_error] > rounded[max_error] else -1
        return rounded


def encode(lattice: Lattice, x, q: int, M: int, eps):
    output = []
    g = x
    for m in range(M):
        g = lattice.nearest_lattice_point(g + eps)
        invG_g = np.dot(lattice.invG, g)
        b_m = invG_g % q
        g = g / q
        output.append(b_m)
    overload_error = not np.isclose(lattice.nearest_lattice_point(g + eps), np.zeros_like(g)).all()
    return output, overload_error


def decode(lattice: Lattice, encoded_vectors, q: int, eps):
    x = np.zeros(lattice.n)

    for m, b_m in enumerate(encoded_vectors):
        x_m = np.dot(lattice.G, b_m) - q * lattice.nearest_lattice_point(np.dot(lattice.G, b_m) / q + eps)
        x += x_m * q ** m
    return x

