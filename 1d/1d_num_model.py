import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import time

def ode(z, y):
    f = y[0] # y1 = f
    df_dz = y[1] # y2 = df/dz
    dy2_dz = -K_sqr * (7/2 * f)**(-3/7)
    return np.array([df_dz, dy2_dz])

def BC(ya, yb):
    return np.array([yb[0] - f_Z, ya[1] - df0_dz])

@nb.jit
def ini_guess(f_t, z_arr, Lz):
    guess_arr = np.ones((2, z_arr.size))
    guess_arr[0, :] = f_t + heat_flux / a * (Lz - z_arr)
    guess_arr[1, :] = heat_flux / (-a)
    return guess_arr

epsilon0 = 8.8541878188e-12
me = 9.1093837015e-31
coulomb_log = 13
Lz = 50.0 # [m]
m = 1.67e-27# [kg] 
gamma_hat = 7.0 * np.sqrt(2/m)
e = 1.60218e-19 # [C]
a = 9.6 * ((2 * np.pi)**(3/2) * epsilon0**2) / (me**(1/2) * e**(1/2) * coulomb_log) # [W/m eV^-7/2]
heat_flux = 1e7 # [W/m2]

N = int(1e2)
z_arr = np.linspace(0, Lz, int(1e2))
K_sqr = 0
Tt_0 = np.logspace(0, 3, N)
ft_0 = (2/7) * Tt_0**(7/2)
Tu_0 = np.zeros(N)
qt_0 = np.zeros(N)
nt_0 = np.zeros(N)
nu_0 = np.zeros(N)
fu_0 = np.zeros(N)

start_time = time.time()
for index in range(N):
    f_Z = 2/7 * Tt_0[index]**(7/2) # f(Z) = f(T_t)
    df0_dz = heat_flux / (-a) # df(Z)/dz = f(q_u)
    y_ini = ini_guess(f_Z, z_arr, Lz)
    solution = solve_bvp(ode, BC, z_arr, y_ini, max_nodes = 1e6, tol = 1e-6)
    fu_0[index] = solution.y[0][0]
    Tu_0[index] = (7/2 * solution.y[0][0])**(2/7)
    qt_0[index] = np.sqrt(K_sqr * a**2 * (Tu_0[index]**2 - Tt_0[index]**2) + heat_flux**2)
    nt_0[index] = qt_0[index] / (gamma_hat * (e * Tt_0[index])**(3/2) * 1e18)
nu_0 = 2*nt_0*Tt_0/Tu_0
end_time = time.time()
print(f'Standard model calculation time = {(end_time - start_time):.6f} s')

N = int(1e2)
K_sqr = 1e6
z_arr = np.linspace(0, Lz, int(1e2))
Tt_arr = np.logspace(0, 3, N)
ft_arr = (2/7) * Tt_arr**(7/2)
Tu_arr = np.zeros(N)
qt_arr = np.zeros(N)
nt_arr = np.zeros(N)
nu_arr = np.zeros(N)
fu_arr = np.zeros(N)

start_time = time.time()
for index in range(N):
    f_Z = 2/7 * Tt_arr[index]**(7/2) # f(Z) = f(T_t)
    df0_dz = heat_flux / (-a) # df(Z)/dz = f(q_u)
    y_ini = ini_guess(f_Z, z_arr, Lz)
    solution = solve_bvp(ode, BC, z_arr, y_ini, max_nodes = 1e6, tol = 1e-4)
    fu_arr[index] = solution.y[0][0]
    Tu_arr[index] = (7/2 * solution.y[0][0])**(2/7)
    qt_arr[index] = np.sqrt(K_sqr * a**2 * (Tu_arr[index]**2 - Tt_arr[index]**2) + heat_flux**2)
    nt_arr[index] = qt_arr[index] / (gamma_hat * (e * Tt_arr[index])**(3/2) * 1e18)
nu_arr = 2*nt_arr*Tt_arr/Tu_arr
end_time = time.time()
print(f'Numerical solution calculation time = {(end_time - start_time):.6f} s')

plt.figure()
plt.title(f'$K^2 = {K_sqr},\ q_u = {heat_flux/1e6} MW/m^2$ \n Dashed line = Standard model)', fontsize = 13)
plt.loglog(Tt_arr, Tu_arr, color = 'b', label = '$T_u$')
plt.loglog(Tt_arr, qt_arr/1e6, color = 'orange', label = '$q_t$')
plt.loglog(Tt_arr, nt_arr, color = 'g', label = '$n_t$')
plt.loglog(Tt_arr, nu_arr, color = 'black', label = '$n_u$')

plt.loglog(Tt_0, nt_0, color = 'g', linestyle = '--')
plt.loglog(Tt_0, Tu_0, color = 'b', linestyle = '--')
plt.loglog(Tt_0, nu_0, color = 'black', linestyle = '--')
plt.loglog(Tt_0, qt_0/1e6, color = 'orange', linestyle = '--')
plt.xlabel('$Target\ Temperature\ T_t$', fontsize = 10)
plt.ylabel('$Temperature\ [eV]$, $Heat\ flux\ [MW/m^2]$ \n $Density\ [10^{18}m^{-3}]$', fontsize = 10)
plt.legend(fontsize = 10)
plt.ylim(1, 1e3)
plt.xlim(1, 1e3)
plt.show()

plt.figure()
plt.title(f'$K^2 = {K_sqr},\ q_u = {heat_flux/1e6} MW/m^2$ \n Dashed line = Standard model)', fontsize = 13)
plt.loglog(nt_arr, Tu_arr, color = 'b', label = '$T_u$')
plt.loglog(nt_arr, qt_arr/1e6, color = 'orange', label = '$q_t$')
plt.loglog(nt_arr, Tt_arr, color = 'r', label = '$T_t$')
plt.loglog(nt_arr, nu_arr, color = 'black', label = '$n_u$')
plt.loglog(nt_0, Tt_0, color = 'r', linestyle = '--')
plt.loglog(nt_0, Tu_0, color = 'b', linestyle = '--')
plt.loglog(nt_0, nu_0, color = 'black', linestyle = '--')
plt.loglog(nt_0, qt_0/1e6, color = 'orange', linestyle = '--')
plt.ylabel('$Temperature\ [eV]$, $Heat\ flux\ [MW/m^2]$ \n $Density\ [10^{18}m^{-3}]$', fontsize = 10)
plt.xlabel('$Target\ Density\ n_t$', fontsize = 10)
plt.legend(fontsize = 10)
plt.ylim(1, 1e3)
plt.xlim(1, 1e3)
plt.show()