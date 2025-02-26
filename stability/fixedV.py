import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar
import time
import warnings
import math

#warnings.filterwarnings("ignore", category=RuntimeWarning)
@nb.jit
def ini_guess(f_t, z_arr, Lz, qu_input, chi): #ode guess
    guess_arr = np.ones((2, z_arr.size))
    guess_arr[0, :] = f_t + qu_input / chi * (Lz - z_arr)
    guess_arr[1, :] = qu_input / (-chi)
    return guess_arr #y_ini = ini_guess(T_t_input, z_arr_input, Lz)

def my_incredible_ode_solver(T_t_input, z_arr_input, initial_guess, K_square):
    @nb.jit
    def BC(ya, yb):
        return np.array([yb[0] - f_Z, ya[1] - df0_dz])
    @nb.jit
    def ode(z, y):
        f = y[0] # y1 = f
        df_dz = y[1] # y2 = df/dz
        dy2_dz = -K_square * (7/2 * f)**(-3/7)
        return [df_dz, dy2_dz]

    f_Z = 2/7 * T_t_input**(7/2) # f(Z) = f(T_t)
    df0_dz = heat_flux / (-chi_hat) # df(Z)/dz = f(q_u)
    solution = solve_bvp(ode, BC, z_arr_input, initial_guess, max_nodes = 1e6, tol = 1e-3)
    (solution.message != 'The algorithm converged to the desired accuracy.') and print(solution.message)
    T_arr_output = (7/2 * solution.y[0])**(2/7)
    dT_arr_output = solution.y[1]/(T_arr_output**(5/2))
    
    qt_arr_output = np.sqrt(K_square * chi_hat**2 * (T_arr_output[0]**2 - T_arr_output[-1]**2) + heat_flux**2)
    voltage_output = (qt_arr_output-heat_flux) / np.sqrt(K_square * chi_hat * sigma_hat)
    return T_arr_output, dT_arr_output, voltage_output, solution.y[0], solution.y[1]

def g(x):
    return my_incredible_ode_solver(Tt, z_space, y_ini, x)[2] - V_expected 

Lz = 50.0 # [m]
m = 1.67e-27# [kg] 
gamma_hat = 7.0 * np.sqrt(2/m)
e = 1.60218e-19 # [C]
heat_flux = 1e7 # [W/m2]
V_expected = 0.1 # [V]
epsilon0 = 8.8541878188e-12
me = 9.1093837015e-31
coulomb_log = 13
chi_hat = 9.6 * ((2 * np.pi)**(3/2) * epsilon0**2) / (me**(1/2) * e**(1/2) * coulomb_log) # [W/m eV^-7/2]
sigma_hat = 0.6125 * chi_hat

N = int(20)
N_nodes = int(1e3)
Tt_arr = np.logspace(0, 3, N)
z_arr = np.linspace(0, Lz, N_nodes)
z_space = z_arr

Tu_arr = np.zeros(N)
qt_arr = np.zeros(N)
nt_arr = np.zeros(N)
nu_arr = np.zeros(N)
K_arr = np.zeros(N)

K_sqr_ini = 0
K_arr[-1] = K_sqr_ini
y_ini = ini_guess(Tt_arr[0], z_arr, Lz, heat_flux, chi_hat)

start_time = time.time()
for i in range(N):
    Tt = Tt_arr[i]
    T_arr = np.zeros((N_nodes))
    dT_arr = np.zeros((N_nodes))
    result = root_scalar(g, x0=K_arr[i-1], method='secant') #root_scalar(g, x0=K_sqr_ini, method='secant')
    if math.isnan(result.root) or (result.root==0):
        print('Break error in index = ', i)
        break
    K_arr[i] = result.root
    T_arr, dT_arr, V_calc, f_arr, df_arr = my_incredible_ode_solver(Tt, z_space, y_ini, result.root)
    y_ini = np.array([f_arr, df_arr])
    z_space = np.linspace(0, Lz, f_arr.size)
    Tu_arr[i] = T_arr[0]
    qt_arr[i] = np.sqrt(K_arr[i] * chi_hat**2 * (Tu_arr[i]**2 - Tt**2) + heat_flux**2)
    nt_arr[i] = qt_arr[i] / (gamma_hat * (e * Tt)**(3/2) * 1e18)
    result.clear()

end_time = time.time()
print(f'Voltage fixed - calculation time = {(end_time - start_time):.6f} s')
nu_arr = 2*nt_arr*Tt_arr/Tu_arr

N = int(20)
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
    Tt = Tt_0[index]
    T_arr, dT_arr, V_calc, f_arr, df_arr = my_incredible_ode_solver(Tt, z_arr, ini_guess(Tt, z_arr, Lz, heat_flux, chi_hat), 0)
    Tu_0[index] = T_arr[0]
    qt_0[index] = np.sqrt(K_sqr * chi_hat**2 * (Tu_0[index]**2 - Tt_0[index]**2) + heat_flux**2)
    nt_0[index] = qt_0[index] / (gamma_hat * (e * Tt_0[index])**(3/2) * 1e18)
nu_0 = 2*nt_0*Tt_0/Tu_0
end_time = time.time()
print(f'Standard model calculation time = {(end_time - start_time):.6f} s')

plt.title(f'$Voltage = {V_expected} V,\ q_u = {heat_flux/1e6} MW/m^2$ \n Dashed line = Standard model', fontsize = 13)
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