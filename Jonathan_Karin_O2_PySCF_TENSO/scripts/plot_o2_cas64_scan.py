#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HARTREE_TO_EV = 27.211386245988
ANGSTROM_TO_M = 1.0e-10
HARTREE_TO_J = 4.3597447222071e-18
ATOMIC_MASS_UNIT = 1.66053906660e-27
SPEED_OF_LIGHT_CM_S = 2.99792458e10

INPUT_FILE = Path("data/o2_cas64_scan.csv")
PLOT_DIR = Path("plots")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_FILE)

    r = data["bond_length_angstrom"].to_numpy()
    e_triplet = data["triplet_energy_hartree"].to_numpy()
    e_singlet = data["singlet_energy_hartree"].to_numpy()
    gap_ev = data["gap_ev"].to_numpy()

    # ---------------------------------------------------------
    # Relative potential-energy curves
    # ---------------------------------------------------------
    triplet_relative_ev = (
        e_triplet - np.min(e_triplet)
    ) * HARTREE_TO_EV

    singlet_relative_ev = (
        e_singlet - np.min(e_singlet)
    ) * HARTREE_TO_EV

    plt.figure(figsize=(7, 5))
    plt.plot(r, triplet_relative_ev, "o-", label="Triplet")
    plt.plot(r, singlet_relative_ev, "s-", label="Singlet")
    plt.xlabel("O-O bond length (Å)")
    plt.ylabel("Energy relative to state minimum (eV)")
    plt.title("O$_2$ CAS(6,4)/cc-pVDZ Potential-Energy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "o2_cas64_potential_curves.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Quadratic fit to the triplet curve near its minimum
    # Use five points centered around the minimum.
    # ---------------------------------------------------------
    minimum_index = int(np.argmin(e_triplet))

    lower = max(0, minimum_index - 2)
    upper = min(len(r), minimum_index + 3)

    r_fit = r[lower:upper]
    e_fit = e_triplet[lower:upper]

    quadratic_coefficients = np.polyfit(r_fit, e_fit, 2)
    a, b, c = quadratic_coefficients

    equilibrium_bond_length = -b / (2.0 * a)
    fitted_minimum_energy = np.polyval(
        quadratic_coefficients,
        equilibrium_bond_length,
    )

    r_dense = np.linspace(r_fit.min(), r_fit.max(), 300)
    fitted_curve = np.polyval(quadratic_coefficients, r_dense)

    plt.figure(figsize=(7, 5))
    plt.plot(
        r,
        (e_triplet - fitted_minimum_energy) * HARTREE_TO_EV,
        "o",
        label="Calculated triplet energies",
    )
    plt.plot(
        r_dense,
        (fitted_curve - fitted_minimum_energy) * HARTREE_TO_EV,
        "-",
        label="Quadratic fit near minimum",
    )
    plt.axvline(
        equilibrium_bond_length,
        linestyle="--",
        label=f"Fitted $R_e$ = {equilibrium_bond_length:.4f} Å",
    )
    plt.xlabel("O-O bond length (Å)")
    plt.ylabel("Relative triplet energy (eV)")
    plt.title("Triplet O$_2$ Equilibrium-Geometry Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "o2_cas64_triplet_quadratic_fit.png", dpi=300)
    plt.close()

    # E = a R^2 + b R + c
    # d2E/dR2 = 2a in Hartree/Angstrom^2
    force_constant_hartree_per_angstrom2 = 2.0 * a

    force_constant_si = (
        force_constant_hartree_per_angstrom2
        * HARTREE_TO_J
        / ANGSTROM_TO_M**2
    )

    oxygen_mass = 15.999 * ATOMIC_MASS_UNIT
    reduced_mass = oxygen_mass / 2.0

    angular_frequency = np.sqrt(force_constant_si / reduced_mass)
    frequency_cm1 = angular_frequency / (
        2.0 * np.pi * SPEED_OF_LIGHT_CM_S
    )

    # ---------------------------------------------------------
    # Linear fit of the singlet-triplet gap near Re
    # ---------------------------------------------------------
    gap_fit = gap_ev[lower:upper]

    slope_ev_per_angstrom, intercept_ev = np.polyfit(
        r_fit,
        gap_fit,
        1,
    )

    gap_at_equilibrium = (
        slope_ev_per_angstrom * equilibrium_bond_length
        + intercept_ev
    )

    gap_dense = (
        slope_ev_per_angstrom * r_dense
        + intercept_ev
    )

    plt.figure(figsize=(7, 5))
    plt.plot(r, gap_ev, "o", label="Calculated CAS(6,4) gap")
    plt.plot(
        r_dense,
        gap_dense,
        "-",
        label="Linear fit near triplet minimum",
    )
    plt.axvline(
        equilibrium_bond_length,
        linestyle="--",
        label=f"$R_e$ = {equilibrium_bond_length:.4f} Å",
    )
    plt.xlabel("O-O bond length (Å)")
    plt.ylabel("Singlet-triplet gap (eV)")
    plt.title("Bond-Length Dependence of the O$_2$ Singlet-Triplet Gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "o2_cas64_singlet_triplet_gap.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Save extracted parameters
    # ---------------------------------------------------------
    output_parameters = pd.DataFrame(
        {
            "parameter": [
                "equilibrium_bond_length_angstrom",
                "triplet_force_constant_N_per_m",
                "harmonic_frequency_cm-1",
                "singlet_triplet_gap_at_Re_eV",
                "gap_slope_eV_per_angstrom",
            ],
            "value": [
                equilibrium_bond_length,
                force_constant_si,
                frequency_cm1,
                gap_at_equilibrium,
                slope_ev_per_angstrom,
            ],
        }
    )

    output_parameters.to_csv(
        "data/o2_cas64_extracted_parameters.csv",
        index=False,
    )

    print("\nExtracted O2 parameters")
    print("=" * 55)
    print(
        f"Equilibrium bond length:     "
        f"{equilibrium_bond_length:.6f} Angstrom"
    )
    print(
        f"Triplet force constant:      "
        f"{force_constant_si:.3f} N/m"
    )
    print(
        f"Harmonic O-O frequency:      "
        f"{frequency_cm1:.2f} cm^-1"
    )
    print(
        f"Gap at equilibrium:          "
        f"{gap_at_equilibrium:.6f} eV"
    )
    print(
        f"d(gap)/dR near equilibrium:  "
        f"{slope_ev_per_angstrom:.6f} eV/Angstrom"
    )
    print("=" * 55)

    print("\nSaved:")
    print("  plots/o2_cas64_potential_curves.png")
    print("  plots/o2_cas64_triplet_quadratic_fit.png")
    print("  plots/o2_cas64_singlet_triplet_gap.png")
    print("  data/o2_cas64_extracted_parameters.csv")


if __name__ == "__main__":
    main()
