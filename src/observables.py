def total_energy(model):
    """
    Compute the total energy of the model’s current spin grid.

    Args:
        model: an IsingModel instance.

    Returns:
        float: Sum of all pairwise interactions (each counted once).
    """
    E = 0.0
    L = model.L
    spins = model.spins

    # Loop over every cell in the L×L grid:
    for i in range(L):
        for j in range(L):
            # Look only “right” and “down” to avoid double-counting each pair:
            right = spins[i, (j + 1) % L]
            down  = spins[(i + 1) % L, j]

            # Interaction energy: -J * s(i,j) * s(neighbor) with J = 1
            E -= spins[i, j] * (right + down)

    return E

def magnetization(model):
    """
    Compute net magnetization: sum of all spins in the grid.

    Args:
        model: an IsingModel instance.

    Returns:
        int: Positive if more +1 spins, negative if more -1 spins.
    """
    return model.spins.sum()
