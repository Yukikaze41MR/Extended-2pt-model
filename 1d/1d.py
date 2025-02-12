import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def ode_1d(z, y, K_sqr):
    T, dTdz = y
    d2Tdz2 = (-K_sqr * T**(-3/2) - 5/2 * T**(3/2) * dTdz**2) / T**(5/2)
    return [dTdz, d2Tdz2]

# Physics param
a = 2.0e3 # [W/m eV^-7/2]
Lz = 50.0 # [m]
m = 1.67e-27 # [kg]
gamma_hat = 7.0*np.sqrt(2/m)
q = 1e7 # [W/m2]
e = 1.60218e-19 # [C]

K_sqr = 0.1
T0 = 1000
dTdz0 = 0.0
y0 = [T0, dTdz0]

solution = solve_ivp(ode_1d, [0, Lz], y0, args=(K_sqr,), dense_output=True)

z_vals = np.linspace(0, Lz, 100)
T_vals = solution.sol(z_vals)[0]

plt.plot(z_vals, T_vals)
plt.xlabel('z')
plt.ylabel('T(z)')
plt.title('Solution of ODE')
plt.show()