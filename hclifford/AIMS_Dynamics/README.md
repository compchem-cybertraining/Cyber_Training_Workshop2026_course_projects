This folder contains all of the files relevant to performing the ab initio multiple spawning dynamics in PySPAWN

* I initiated 50 trajectories from the 50 initial conditions i generated and each trajectory generates and/or used 9 files (start.py, start.log, hessian.hdf5, geometry.xyz, sim.hdf5, sim.json, sim2.hdf5, and sim2.json, Slurm.job). This is a very large amount of files so i have left initial condition #1 as is for an example but i have zipped IC 1 to 50 into a zip file (IC_1_50.zip) only including start.py, start.log, and geometry.xyz and uploaded here.




Included is the mpg file of an example trajectory across the 750 fs. There are also 2 png of the plots of dynamics data

IC_List.txt has the list of the random seeds used to generate each initial condition

There are individual directories for each initial condition

hessian.hdf5 is the Hessian used for Wigner sampling of positions and momenta

start.py is the input file of the trajectory. It has all of the dynamics set up details such as timestep, legnth of trajectory, and other details of the calculation

start.log is the output file

sim.hdf5, sim.json, sim2.hdf5, and sim2.json are files generated in PySPAWN that store info about the AIMS wave function, energies, forces, position and momenta of each trajectory basis function, and other quantities

geometry.xyz is the xyz coordinates of the initial condition
