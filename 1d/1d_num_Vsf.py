import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.integrate import simpson
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

#@nb.jit
def integration(x, y):
    delta_x = np.diff(x)
    #return np.trapz(y, delta_x)
    #return simpson(y = y, x = delta_x) # Slower
    return np.sum((y[:-1] + y[1:]) / 2 * delta_x)

epsilon0 = 8.8541878188e-12
me = 9.1093837015e-31
coulomb_log = 13
a = 9.6 * ((2 * np.pi)**(3/2) * epsilon0**2) / (me**(1/2) * e**(1/2) * coulomb_log) # [W/m eV^-7/2]
Lz = 50.0 # [m]
m = 1.67e-27 # [kg] (Deuterium: 3.34358372e-27)
m_i = m
m_e = 9.10938356e-31 # [kg]
gamma_hat = 7.0 * np.sqrt(2/m)
e = 1.60218e-19 # [C]
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
nu_arr = 2*nt_0*Tt_0/Tu_0
V_sf0 = 1/2 * np.log(2* np.pi* m_e/m_i * (1 + 1)) * Tt_0 #* kT_e[K]/e
end_time = time.time()
print(f'Standard model calculation time = {(end_time - start_time):.6f} s')

N = int(1e2)
K_sqr = 1e6
z_arr = np.linspace(0, Lz, int(1e4))
Tt_arr = np.logspace(0, 3, N)
ft_arr = (2/7) * Tt_arr**(7/2)
delta_z = np.diff(z_arr)
T_arr = np.zeros(N)
R_arr = np.zeros(N)
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
    solution = solve_bvp(ode, BC, z_arr, y_ini, max_nodes = 1e6, tol = 1e-3)
    z_arr = np.linspace(0, Lz, solution.x.size)
    R_arr[index] = integration(z_arr, ((7/2 * solution.y[0])**(2/7))**(-3/2))
    fu_arr[index] = solution.y[0][0]
    Tu_arr[index] = (7/2 * solution.y[0][0])**(2/7)
    qt_arr[index] = np.sqrt(K_sqr * a**2 * (Tu_arr[index]**2 - Tt_arr[index]**2) + heat_flux**2)
    nt_arr[index] = qt_arr[index] / (gamma_hat * (e * Tt_arr[index])**(3/2) * 1e18)
nu_arr = 2*nt_arr*Tt_arr/Tu_arr
V_sf = 1/2 * np.log(2* np.pi* m_e/m_i * (1 + 1)) * Tt_0 #* kT_e[K]/e
end_time = time.time()
print(f'Numerical solution calculation time = {(end_time - start_time):.6f} s')

sigma_hat = a * 0.6125
deltaq_arr = qt_arr - heat_flux
R_arr = R_arr / sigma_hat