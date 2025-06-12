# run_simulation.py
# ------------------------------
print("run_simulation.py loaded")  # ── ADDED: sanity check at import
print("Module __name__ is:", __name__)  # ── ADDED: confirm main guard will run

import numpy as np
import matplotlib.pyplot as plt
import time  # ── ADDED: timing module
from ising import IsingModel
from observables import total_energy, magnetization


def simulate(L_values, temps, n_eq, n_samp):
    """
    Run MC across lattice sizes; compute heat capacity & Binder cumulant.
    """
    results = {}
    for L in L_values:  # for each size
        print(f"\n→ Starting simulations for L={L}")  # ADDED: clear newline per-L
        heat_caps, mags, binders = [], [], []
        for idx, T in enumerate(temps):
            # THROTTLED: print every 5 steps or the last step
            if (idx + 1) % 5 == 0 or idx == len(temps) - 1:
                progress = f"→ L={L}, T={T:.2f} ({idx+1}/{len(temps)})"
                print(progress, end="\r")  # ADJUSTED: removed flush for fewer prints

            model = IsingModel(L, T)
            # Equilibration
            for _ in range(n_eq):
                model.metropolis_step()  # CHANGED: uses JIT-compiled sweep inside
            # Sampling
            E_acc = E2_acc = 0.0
            M_acc = M2_acc = M4_acc = 0.0
            for _ in range(n_samp):
                model.metropolis_step()  # CHANGED: JIT jump drastically speeds updates
                E = total_energy(model)
                m = magnetization(model)
                E_acc += E
                E2_acc += E * E
                M_acc += abs(m)
                M2_acc += m * m
                M4_acc += m ** 4
            # Compute observables
            E_mean = E_acc / n_samp
            E2_mean = E2_acc / n_samp
            C = (E2_mean - E_mean**2) / (T**2 * L * L)
            M2_mean = M2_acc / n_samp
            M4_mean = M4_acc / n_samp
            U = 1 - M4_mean / (3 * M2_mean**2)
            heat_caps.append(C)
            mags.append(M_acc / (n_samp * L * L))
            binders.append(U)
        print()  # ADDED: newline after finishing all temperatures for this L
        results[L] = {
            'heat_capacity': np.array(heat_caps),
            'magnetization': np.array(mags),
            'binder': np.array(binders)
        }
    return results


if __name__ == '__main__':
    print("▶ Starting simulation…")  # ── ADDED: beginning notice
    L_values = [32, 48, 64]  # sizes for finite-size scaling
    temps = np.linspace(1.5, 3.5, 21)
    n_eq, n_samp = 1000, 5000

    t0 = time.time()  # ── ADDED: start time
    results = simulate(L_values, temps, n_eq, n_samp)
    elapsed = time.time() - t0  # ── ADDED: elapsed time

    print(f"\n✅ Simulation finished in {elapsed:.2f} s; now plotting.")  # ── ADDED: summary timing

    # Save data
    np.savez(
        'ising_results.npz',
        temps=temps,
        **{f'C_L{L}': results[L]['heat_capacity'] for L in L_values},
        **{f'U_L{L}': results[L]['binder'] for L in L_values}
    )

    # Plot heat capacity
    plt.figure()
    for L in L_values:
        plt.plot(temps, results[L]['heat_capacity'], marker='o', label=f"L={L}")
    plt.xlabel('Temperature (T)')
    plt.ylabel('Heat Capacity per spin')
    plt.title('Heat Capacity vs Temperature')
    plt.legend()
    plt.show()

    # Plot Binder cumulant
    plt.figure()
    for L in L_values:
        plt.plot(temps, results[L]['binder'], marker='o', label=f"L={L}")
    plt.xlabel('Temperature (T)')
    plt.ylabel('Binder Cumulant')
    plt.title('Binder Cumulant vs Temperature')
    plt.legend()
    plt.show()

    print(" Done! Energy and Binder cumulant plots displayed.")  # ── ADDED: final confirmation