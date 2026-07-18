#!/usr/bin/env python3
"""
Compute a Drude-Lorentz + Brownian Oscillator spectral density.

TENSO conventions:
  J_DL(ω) = (2 λ / π) * (ω_c ω) / (ω^2 + ω_c^2)
  J_BO(ω) = (4 λ / π) * η * ω0^2 * ω / ((ω^2 - ω0^2)^2 + 4 η^2 ω^2)

All frequencies must use the same units.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def print_banner():
    banner = r"""
   ███████╗██████╗ ███████╗ ██████╗
   ██╔════╝██╔══██╗██╔════╝██╔════╝
   ███████╗██████╔╝█████╗  ██║
   ╚════██║██╔═══╝ ██╔══╝  ██║
   ███████║██║     ███████╗╚██████╗
   ╚══════╝╚═╝     ╚══════╝ ╚═════╝

   ██████╗ ███████╗███╗   ██╗███████╗
   ██╔══██╗██╔════╝████╗  ██║██╔════╝
   ██║  ██║█████╗  ██╔██╗ ██║███████╗
   ██║  ██║██╔══╝  ██║╚██╗██║╚════██║
   ██████╔╝███████╗██║ ╚████║███████║
   ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝

         A TENSO-like spectral 
           density generator
    """
    print(banner)


def j_drude_lorentz(w, re_d, width_d):
    """Drude-Lorentz spectral density."""
    w = np.asarray(w, dtype=float)
    return (2.0 * re_d / np.pi) * (width_d * w) / (w**2 + width_d**2)


def j_brownian_oscillator(w, freq_b, re_b, width_b):
    """Underdamped Brownian oscillator spectral density."""
    w = np.asarray(w, dtype=float)
    return (
        (4.0 * re_b / np.pi)
        * width_b
        * (freq_b**2)
        * w
        / (((w**2 - freq_b**2) ** 2) + 4.0 * (width_b**2) * (w**2))
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute Drude-Lorentz and Brownian oscillator spectral densities."
    )
    parser.add_argument("--re_d", type=float, required=True, help="Drude reorganization energy")
    parser.add_argument("--width_d", type=float, required=True, help="Drude cutoff frequency")
    parser.add_argument("--freq_b", type=float, required=True, help="Brownian oscillator frequency")
    parser.add_argument("--re_b", type=float, required=True, help="Brownian reorganization energy")
    parser.add_argument("--width_b", type=float, required=True, help="Brownian broadening")
    parser.add_argument("--wmin", type=float, default=0.0, help="Minimum frequency")
    parser.add_argument("--wmax", type=float, required=True, help="Maximum frequency")
    parser.add_argument("--npts", type=int, default=2000, help="Number of grid points")
    parser.add_argument("--outfile", default="spectral_density.dat", help="Output data file")
    parser.add_argument("--plot", action="store_true", help="Plot the result")
    args = parser.parse_args()

    w = np.linspace(args.wmin, args.wmax, args.npts)

    jdl = j_drude_lorentz(w, args.re_d, args.width_d)
    jbo = j_brownian_oscillator(w, args.freq_b, args.re_b, args.width_b)
    jtot = jdl + jbo

    np.savetxt(
        args.outfile,
        np.column_stack([w,jtot, jdl, jbo]),
        header="w J_total  J_DrudeLorentz  J_BrownianOscillator",
    )

    print(f"Wrote {args.outfile}")

    if args.plot:
        plt.plot(w, jdl, label="Drude-Lorentz")
        plt.plot(w, jbo, label="Brownian oscillator")
        plt.plot(w, jtot, label="Total", linewidth=2)
        plt.xlabel("Frequency")
        plt.ylabel("J(ω)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print_banner()  
    main()
