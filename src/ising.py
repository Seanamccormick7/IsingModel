# ising.py
# ------------------------------
import numpy as np
from numba import njit  # ADDED: import Numba for JIT compilation

@njit
def _sweep_jit(spins, T, J):
    """
    JIT-compiled Monte Carlo sweep: loops at machine speed.
    """  # ADDED: escape Python overhead by compiling the inner loop
    L = spins.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        # Periodic boundary neighbors
        nb = (spins[(i - 1) % L, j] +
              spins[(i + 1) % L, j] +
              spins[i, (j - 1) % L] +
              spins[i, (j + 1) % L])
        dE = 2 * J * s * nb
        # Use NumPy's rand inside JIT for speed
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] = -s
    return spins

class IsingModel:
    """
    A 2D Ising model on an L×L grid at temperature T, with adjustable coupling J.
    Implements the Metropolis algorithm for Monte Carlo sampling.
    """

    def __init__(self, size, temperature, coupling=1.0):  # CHANGED: added `coupling` parameter
        """
        Args:
            size (int): Grid dimension L.
            temperature (float): System temperature T.
            coupling (float): Interaction strength J (default=1.0).  # ADDED
        """
        self.L = size
        self.T = temperature
        self.J = coupling  # ADDED: store interaction strength
        # Initialize spins randomly to +1 or -1 (unchanged)
        self.spins = np.random.choice([-1, 1], size=(self.L, self.L))

    def _energy_change(self, i, j):
        """
        Compute ΔE for flipping spin at (i,j):
        ΔE = 2 * J * s_ij * (sum of four neighbors).
        """
        s = self.spins[i, j]
        neighbors = (
            self.spins[(i - 1) % self.L, j] +
            self.spins[(i + 1) % self.L, j] +
            self.spins[i, (j - 1) % self.L] +
            self.spins[i, (j + 1) % self.L]
        )
        return 2 * self.J * s * neighbors  # CHANGED: multiply by self.J

    def metropolis_step(self):
        """
        Perform one Monte Carlo sweep (L^2 flip attempts).  # unchanged
        """
        # CHANGED: delegate to JIT-compiled sweep function for speed
        self.spins = _sweep_jit(self.spins, self.T, self.J)
