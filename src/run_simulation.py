# run_simulation.py

print(" run_simulation.py loaded")                      # ── ADDED: sanity check at import
print("Module __name__ is:", __name__)                   # ── ADDED: confirm main guard will run

from ising import IsingModel
from observables import total_energy, magnetization
import numpy as np
import matplotlib.pyplot as plt

def simulate(L, temps, n_eq, n_samp):
    """
    Run simulations over a range of temperatures.

    Args:
        L (int): Grid size (L×L).
        temps (iterable): Temperatures to test.
        n_eq (int): Number of sweeps to equilibrate (no data collection).
        n_samp (int): Number of sweeps to sample/average after equilibration.

    Returns:
        (np.ndarray, np.ndarray):
            energies_per_spin, magnetizations_per_spin
            for each temperature in temps.
    """
    energies = []
    mags     = []

    for i, T in enumerate(temps, start=1):
        # ── ADDED: progress print for each temperature
        print(f"→ Running T = {T:.2f} ({i}/{len(temps)})", end="\r", flush=True)

        # 1) Initialize a fresh IsingModel at temp T:
        model = IsingModel(L, T)

        # 2) Equilibration phase (no recording):
        for _ in range(n_eq):
            model.metropolis_step()

        # 3) Sampling phase (record energy & magnetization):
        E_accum = 0.0
        M_accum = 0.0
        for _ in range(n_samp):
            model.metropolis_step()
            E_accum += total_energy(model)
            M_accum += abs(magnetization(model))

        # 4) Normalize to per‐spin averages:
        norm = 1.0 / (n_samp * L * L)
        energies.append(E_accum * norm)
        mags.append(M_accum * norm)

    return np.array(energies), np.array(mags)

if __name__ == "__main__":
    # ── ADDED: print at start of main
    print("▶ Starting simulation…")

    # Default parameters:
    L      = 20                          # 20×20 lattice
    # ── ADDED: quick‐test override (uncomment for fast sanity checks)
    # L, n_eq, n_samp = 10, 100, 100

    temps  = np.linspace(1.0, 3.5, 25)   # 25 temps from 1.0 to 3.5
    # ── ADDED: quick‐test override
    # temps = np.linspace(2.0, 2.5, 5)

    n_eq   = 1000                        # Sweeps before sampling
    n_samp = 2000                        # Sweeps we sample

    # Run the simulation:
    energies, mags = simulate(L, temps, n_eq, n_samp)

    # ── ADDED: print when simulation ends, before plotting
    print("\n✅ Simulation finished; now plotting.")

    # Plot Energy vs Temperature:
    plt.figure()
    plt.plot(temps, energies, marker='o')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Energy per spin')
    plt.title('Ising Model: Energy vs T')

    # Plot Magnetization vs Temperature:
    plt.figure()
    plt.plot(temps, mags, marker='o')
    plt.xlabel('Temperature (T)')
    plt.ylabel('|Magnetization| per spin')
    plt.title('Ising Model: Magnetization vs T')

    # ── ADDED: final confirmation after plots are displayed
    print(" Done! Energy and Magnetization plots displayed.")

    # Display both plots on screen:
    plt.show()


