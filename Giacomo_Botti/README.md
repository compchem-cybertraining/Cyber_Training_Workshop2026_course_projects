# Giacomo Botti's Project

Hi! Welcome to my project folder for the CyberTraining 2026. 

In this README, I will only provide a short outline of my project and some results. If you want the **real deal**, please read the `report.pdf` file.

## Project Outline

My work is split in two. The first part is creating a quasiclassical
trajectory (QCT) integrator using PySCF, and test it to obtain the
quasiclassical *ab initio* spectrum of $H_{2}$. The second part will consist in
computing the open quantum system dynamics of $H_{2}\cdot C_{60}$, to check if it is possible to compute the spectral density from an _ab initio_ trajectory. I will compute the $S_{0}$ and $S_{1}$ harmonic frequencies for isolated $H_{2}$
using OpenMolcas (this is an approximation, but it will save me time). I will then
move to compute the spectral density of by Fourier-transforming the QM
energy fluctuations in an ONIOM QCT dynamics. The theoretical background
is described in [this
tutorial](https://github.com/compchem-cybertraining/Tutorials_Libra/blob/master/11_program_specific_methods/3_cp2k_methods/3_time_resolved_energies/tutorial.ipynb)
(hoping that the fluctuations of the QM energy are suitable for this task). The ONIOM
simulation will be carried out with  ORCA and
[Dragonball](https://dragonball-vispec.readthedocs.io/en/latest/), so that it can run
in parallel to the other parts of the project. In case the spectral
density could not be computed as expected, the vibrational density of
states (VDOS) of $C_{60}$ would be used instead (this should not be a great
approximation, since VDOS is already pretty simple). As for the
system-bath coupling, some tuning would be necessary, if time allows and
the results are interesting. With the harmonic frequencies and some
spectral density peaks, I can build the model Hamiltonian in and run a
small simulation.

## Conclusions

This projects confirms the old saying that "an hour in the library saves
a week in the lab". Indeed, the aim of this project is to study if a
workflow developed for computing VDOS can be extended to construct
effective spectral density. However, the model system I tried to create
is not at all a good benchmark. First of all, $H_{2}$ and $C_{60}$ do not seem coupled,
even if the stretching is heavily red-shifted. This prevented me to
properly investigate if the QM energy fluctuations are a good way to
analyze the system-bath coupling. Second, $H_{2}$ is probably the worst system
to study vibronic transitions, given the energies at play. This forced me
in the end to use an *ansatz* Hamiltonian part informed, part guessed,
and all boring.

As for the workflow, PySCF is a good starting point to improve some of the
issues I had with ORCA and OpenMolcas. First
of all, from ORCA outputs I was able to recover only the
total ONIOM energy or the QM1. This again stresses the importance of
having greater control over the *ab initio* outputs. As for OpenMolcas, a PySCF script
would have probably handled the surface scan much better. This time I
was lucky that I was working with $H_{2}$. As anticipated, I will continue
working on `runner.py`. First, I would like to include an Adiabatic Switching
version, that requires to slowly changing the Hamiltonian from fully
harmonic to fully *ab initio*. Then I would love to use flexibility to
use other *ab initio* packages to compute the forces. Finally, I think
it would be interesting to find out what a quasiclassical VDOS of an
excited state could tell us.

As for the results, they are fully consistent with the model
Hamiltonian. Nonetheless, the aim of this project was obtaining some
results, not a breakthrough, and I am satisfied with what I got.
Computing the spectral density and, more in general, the coupling
picture from an *ab initio* trajectory remains an open problem, worthy
of investigation.

## Contents of the folder

- The actual report, `report.pdf`; this file is the usual "lab report" that I
  keep for myself, just more curated than usual; I hope it was an
  enjoyable read
- In `/runner/`, I include the input (`geometry.xyz` and `velocity.xyz`) and output (`traj.xyz`,`forces.dat` and `md.log`) files for $H_{2}$ (`/h2/`) and $H_{2}O$ (`/h2o/`); in `/h2o/`, I also include the input necessary to compute the Heller Frozen Gaussian VDOS shown in `report.pdf`

- In `/orca/`, I include the initial guess geometry, and input and relevant output files for  ORCA optimization and frequencies job (`opt-freq`) and _ab initio_ molecular dynamics (`dyn`); the files are divided in three folder based on the system: $C_{60}$ (`/C60_XTB/`), $H_{2}\cdot C_{60}$ (`H2atC60_ONIOM`) and $H_{2}$ (`H2_ORCA`).

- In`/dragonball/`, I include the final output files of the Dragonball runs, for $C_{60}$ (`/c60_xtb/`), $H_{2}\cdot C_{60}$ (`h2atc60`) and $H_{2}$ (`h2_orca`).

- In `/openmolcas/`, I include the input and relevant output files for the $S_{0}$ optimization and frequency run (`H2_S0_optfreq`), the $S_{1}$ optimization and frequency run (`H2_S1_optfreq`), and the surface scan (`H2_avoided`).

- In `/plot_specden/`, I include the script I used to compute the spectral densities (`sd.py`), its output files (`*.dat`) and the final plots (`*.pdf`).

- In `/tenso/`, I include the final run input (`final_run.py`) and
  output (`output.dat.log`), together with the modified plotting script (`plot_tenso.py`) and the two  output plots ( `s0.pdf` and `s1.pdf`).


