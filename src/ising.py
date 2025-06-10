import numpy as np
# Import NumPy under the alias `np`, giving us fast, vectorized arrays.

class IsingModel:
    """
    A 2D Ising model on an L×L grid at temperature T.
    Implements the Metropolis algorithm for Monte Carlo sampling.
    """

    def __init__(self, size, temperature):
        """
        Constructor runs when you call IsingModel(size, temperature).

        Args:
            size (int): Number of rows/columns (grid is size×size).
            temperature (float): Thermal noise level; higher T → more random flips.
        """
        self.L = size
        # Save the grid dimension so other methods know how many spins exist.

        self.T = temperature
        # Save the temperature; used in the Boltzmann acceptance test.

        # Create an L×L array of spins, each randomly +1 or -1:
        self.spins = np.random.choice(
            [-1, 1],       # possible spin values (down, up)
            (self.L, self.L)  # shape of the grid
        )

    def _energy_change(self, i, j):
        """
        Calculate ΔE if we flip the spin at position (i, j).

        Uses periodic boundaries so edges wrap around like a torus.

        Args:
            i (int): Row index of the spin.
            j (int): Column index of the spin.

        Returns:
            float: ΔE = E_new − E_old for flipping that spin.
        """
        spin = self.spins[i, j]
        # Current spin value (+1 or -1).

        # Sum four neighbors (down, up, right, left), using modulo for wrap:
        neighbors = (
            self.spins[(i + 1) % self.L, j] +
            self.spins[(i - 1) % self.L, j] +
            self.spins[i, (j + 1) % self.L] +
            self.spins[i, (j - 1) % self.L]
        )

        # ΔE formula for Ising with coupling J = 1:
        # ΔE = 2 * J * spin * sum_of_neighbor_spins
        delta_E = 2 * spin * neighbors
        return delta_E

    def metropolis_step(self):
        """
        Perform one full sweep (L² attempts) of the Metropolis algorithm:

        For each attempt:
         1. Pick a random spin.
         2. Compute ΔE if flipped.
         3. If ΔE ≤ 0 → accept flip.
         4. Else accept with probability exp(−ΔE / T).
        """
        for _ in range(self.L * self.L):
            # 1) Choose a random site (i, j):
            i = np.random.randint(0, self.L)
            j = np.random.randint(0, self.L)

            # 2) Compute the energy change for flipping it:
            delta_E = self._energy_change(i, j)

            # 3) Decide to flip:
            #    - Always if ΔE ≤ 0 (energy-lowering).
            #    - Otherwise with probability exp(−ΔE / T).
            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / self.T):
                self.spins[i, j] *= -1
                # Multiply by -1 flips +1 ↔ -1.
