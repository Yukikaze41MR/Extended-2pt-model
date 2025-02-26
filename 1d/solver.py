import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import time

@nb.jit
def ode(z, y):
    f = y[0] # y1 = f
    df_dz = y[1] # y2 = df/dz
    dy2_dz = -K_sqr * (7/2 * f)**(-3/7)
    return [df_dz, dy2_dz]

@nb.jit
def BC(ya, yb):
    return np.array([yb[0] - f_Z, ya[1] - df0_dz])

@nb.jit
def ini_guess(f_t, z_arr, Lz): #Linear guess
    guess_arr = np.ones((2, z_arr.size))
    guess_arr[0, :] = f_t + heat_flux / a * (Lz - z_arr)
    guess_arr[1, :] = heat_flux / (-a)
    return guess_arr

N = int(1e4)
z_arr = np.linspace(0, Lz, int(1e4))
T_arr = np.zeros((N))
dT_arr = np.zeros((N))
index = -1

start_time = time.time()
f_Z = 2/7 * Tt[index]**(7/2) # f(Z) = f(T_t)
df0_dz = heat_flux / (-a) # df(Z)/dz = f(q_u)
y_ini = ini_guess(Tt[index], z_arr, Lz)
solution = solve_bvp(ode, BC, z_arr, y_ini, max_nodes = 1e6, tol = 1e-3)

T_arr = (7/2 * solution.y[0])**(2/7)
dT_arr = solution.y[1]/(T_arr**(5/2))
end_time = time.time()
print(f'Numerical solution calculation time = {(end_time - start_time):.6f} s')
print(solution.message, 'Used nodes = ', solution.x.size)

print('Check Neumann BC (f_prime(0)): Error = ', abs(solution.y[1][0] - df0_dz))
print('Check Dirichlet BC (f(L)): Error = ', abs(solution.y[0][-1] - f_Z))

print('Dinne model:', 'T_u = ', Tu[index], ' T_t = ', Tt[index])
print('ODE Solution:', 'T_u = ', T_arr[0], ' T_t = ', T_arr[-1])

#plt.subplot(1, 2, 1)
plt.plot(np.linspace(0, Lz, solution.x.size), dT_arr, color = 'b', label = 'dT/dz (num)')
plt.xlim(0, Lz)
plt.xlabel('z [m]', fontsize = 10)
plt.legend(fontsize = 10)

#plt.subplot(1, 2, 2)
plt.figure()
plt.title(f'$K^2 = {K_sqr},\ q_\parallel = {heat_flux/1e6} MW/m^2$', fontsize = 13)
plt.plot(z_arr, (7/2 * y_ini[0])**(2/7), color = 'b', linestyle = '--', label = 'K = 0')
plt.plot([0, Lz], [Tu[index], Tu[index]], '--', color = 'black', label = '$T_u\ in\ Dinne\ model$')

plt.plot(np.linspace(0, Lz, solution.x.size), T_arr, color = 'r', label = 'T (num)')
plt.xlabel('z [m]', fontsize = 10)
plt.ylabel('Temperature [eV]', fontsize = 10)
#plt.ylabel('f', fontsize = 10)
plt.legend(fontsize = 10)
plt.ylim(Tt[index],)
plt.xlim(0, Lz)
plt.tight_layout()
plt.show()