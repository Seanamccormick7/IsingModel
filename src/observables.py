# observables.py
# ------------------------------
# --- Original observables.py code (commented out) ---
# def total_energy(model):
#     # original computed E ignoring coupling J
#     pass
# def magnetization(model):
#     # original sum of spins
#     pass
# --- End original ---

import numpy as np

def total_energy(model):
    """
    Compute total energy: E = -J * sum_{<ij>} s_i s_j.
    """
    E = 0.0
    L = model.L
    spins = model.spins
    J = model.J  # CHANGED: now includes model.J
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            # Only right and down neighbors to avoid double counting (unchanged)
            E -= J * s * spins[i, (j + 1) % L]
            E -= J * s * spins[(i + 1) % L, j]
    return E


def magnetization(model):  # unchanged signature
    """
    Compute net magnetization: sum of all spins.
    """
    return model.spins.sum()  # unchanged
