import matplotlib.pyplot as plt
import numpy as np

out = 'simple_2000'
results = np.loadtxt(out + ".dat.log", dtype=np.complex128)
plt.plot(np.real(results[:, 0]), np.real(results[:, 1]), label=r"$\rho_{00}$")
plt.plot(np.real(results[:, 0]), np.real(results[:, 4]), label=r"$\rho_{11}$")
plt.plot(np.real(results[:, 0]), np.real(results[:, 9]), label=r"$\rho_{22}$")
plt.legend()
plt.xlabel("Time (fs)")
plt.xlim(0, 200)
plt.ylim(-0.1, 1)
plt.ylabel("Population")
plt.savefig(out + '.png')
plt.show()