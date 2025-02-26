import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def my_incredible_ode_solver(T_t_input, z_arr_input, K_square):
    def BC(ya, yb):
        return np.array([yb[0] - f_Z, ya[1] - df0_dz])
    def ode(z, y):
        f = y[0] # y1 = f
        df_dz = y[1] # y2 = df/dz
        dy2_dz = -K_square * (7/2 * f)**(-3/7)
        return [df_dz, dy2_dz]
    @nb.jit
    def ini_guess(f_t, z_arr, Lz): #Linear guess
        guess_arr = np.ones((2, z_arr.size))
        guess_arr[0, :] = f_t + heat_flux / chi_hat * (Lz - z_arr)
        guess_arr[1, :] = heat_flux / (-chi_hat)
        return guess_arr

    f_Z = 2/7 * T_t_input**(7/2) # f(Z) = f(T_t)
    df0_dz = heat_flux / (-chi_hat) # df(Z)/dz = f(q_u)
    y_ini = ini_guess(T_t_input, z_arr_input, Lz)
    solution = solve_bvp(ode, BC, z_arr_input, y_ini, max_nodes = 1e6, tol = 1e-5)
    f_arr_output = solution.y[0]
    T_arr_output = (7/2 * solution.y[0])**(2/7)
    dT_arr_output = solution.y[1]/(T_arr_output**(5/2))
    
    qt_arr_output = np.sqrt(K_square * chi_hat**2 * (T_arr_output[0]**2 - T_arr_output[-1]**2) + heat_flux**2)
    voltage_output = (qt_arr_output-heat_flux) / np.sqrt(K_square * chi_hat * sigma_hat)
    return T_arr_output, dT_arr_output, qt_arr_output, voltage_output, f_arr_output

def g(x):
    return my_incredible_ode_solver(Tt, z_arr, x)[3] - V_expected 

N = int(1e4)
K_sqr_ini = 942784167.6743213
chi_hat = 2.0e3 # [W/m eV^-7/2]
Lz = 50.0 # [m]
m = 1.67e-27# [kg] 
e = 1.60218e-19 # [C]
heat_flux = 2848035868.435805 # [W/m2]
V_expected = 350 # [V]

epsilon0 = 8.8541878188e-12
me = 9.1093837015e-31
coulomb_log = 13
chi_hat = 9.6 * ((2 * np.pi)**(3/2) * epsilon0**2) / (me**(1/2) * e**(1/2) * coulomb_log) # [W/m eV^-7/2]
sigma_hat = 0.6125 * chi_hat

Tt = 200
z_arr = np.linspace(0, Lz, N)
T_arr = np.zeros((N))
f_arr = np.zeros((N))
dT_arr = np.zeros((N))
print('Initial calculated voltage = ', my_incredible_ode_solver(Tt, z_arr, K_sqr_ini)[3])

start_time = time.time()
result = root_scalar(g, x0=K_sqr_ini, method='secant') #root_scalar(g, x0=K_sqr_ini, method='secant')
end_time = time.time()
print(f'Optimization time = {(end_time - start_time):.6f} s')
print('K^2 value =', result.root)
print('Expected voltage = ', V_expected)
T_arr, dT_arr, qt, V_calc, f_arr = my_incredible_ode_solver(Tt, z_arr, result.root)

print('Calculated voltage = ', V_calc)

plt.figure()
plt.title(f'$Voltage = {V_expected} V, K^2 = {result.root},\ q_\parallel = {heat_flux/1e6} MW/m^2$', fontsize = 13)
plt.plot(np.linspace(0, Lz, T_arr.size), T_arr, color = 'r', label = 'T (num)')
plt.xlabel('z [m]', fontsize = 10)
plt.ylabel('Temperature [eV]', fontsize = 10)
#plt.ylabel('f', fontsize = 10)
plt.legend(fontsize = 10)
#plt.ylim(Tt,)
plt.xlim(0, Lz)
plt.tight_layout()
plt.show()

def perturbation_ode_solver(f0_input, z_arr_input, Ksqr_input, bc1 = 1, bc2 = 1):
    def BC_perturbation(ya, yb):
        return np.array([ya[0] - bc1, yb[0] - bc2])
    def ode_perturbation(z, y):
        f1 = y[0] # y1 = f
        df_dz = y[1] # y2 = df/dz
        f0 = np.interp(np.linspace(0, 1, f1.size), np.linspace(0, 1, f0_input.size), f0_input, left=f0_input[0], right=f0_input[-1])
        df2_dz = 3/2 * Ksqr_input * (7/2)**(-10/7) * f0**(-10/7) * f1
        return [df_dz, df2_dz]
    @nb.jit
    def ini_guess(value1, slope, z): #Linear guess
        guess_arr = np.ones((2, z.size))
        guess_arr[0] = value1 + slope * z
        guess_arr[1] = slope
        return guess_arr
    
    y_ini = ini_guess(bc1, (bc2-bc1)/Lz, z_arr_input)
    solution = solve_bvp(ode_perturbation, BC_perturbation, z_arr_input, y_ini, max_nodes = 1e6, tol = 1e-3)
    print(solution.message)
    return solution.y[0], solution.y[1]

j0 = np.sqrt(result.root * chi_hat * sigma_hat)
z_arr = np.linspace(0, Lz, f_arr.size)

T_plus, dT_plus = perturbation_ode_solver(f_arr, z_arr, result.root, 1 * T_arr[0]**(5/2), 1 * T_arr[-1]**(5/2))
T_minus, dT_minus = perturbation_ode_solver(f_arr, z_arr, result.root, 1 * T_arr[0]**(5/2), -1 * T_arr[-1]**(5/2))

T_plus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, T_plus.size), T_plus)
T_minus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, T_minus.size), T_minus)
dT_plus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, dT_plus.size), dT_plus)
dT_minus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, dT_minus.size), dT_minus)

Wronskian = abs(T_plus * dT_minus - T_minus * dT_plus) #np.round(Wronskian, decimals = 16)
plt.figure()
plt.title(f'Wronskian $f_+df_- - f_-df_+$')
plt.plot(np.linspace(0, Lz, T_arr.size), np.round(Wronskian, decimals = 5), color = 'orange')
plt.ylim(np.min(Wronskian)-1, np.min(Wronskian)+1)
plt.xlim(0, Lz)

T_plus = T_plus / T_arr**(5/2)
T_minus = T_minus / T_arr**(5/2)
plt.figure()
plt.title(f'$Voltage = {np.round(V_calc, decimals = 2)} V,\ q_\parallel = {heat_flux/1e6} MW/m^2$', fontsize = 13)
plt.plot(np.linspace(0, Lz, T_plus.size), T_plus, color = 'b', label = '$T_+$')
plt.plot(np.linspace(0, Lz, T_minus.size), T_minus, color = 'r', label = '$T_-$')
plt.xlabel('z [m]', fontsize = 10)
plt.ylabel('Temperature [eV]', fontsize = 10)
#plt.ylabel('f', fontsize = 10)
plt.legend(fontsize = 10)
#plt.ylim(Tt,)
plt.xlim(0, Lz)
plt.tight_layout()
plt.show()

from scipy.integrate import simpson

M11 = 7/2 * heat_flux;      M12 = -chi_hat * dT_plus[0];        M13 = -chi_hat * dT_minus[0]
M21 = qt + 5/2 * heat_flux;     M22 = -chi_hat * dT_plus[-1];       M23 = -chi_hat * dT_minus[-1]
M31 = 2 * qt - 5/2 * heat_flux;     M32 = - 1/2 * qt * T_plus[-1]/T_arr[-1];        M33 = - 1/2 * qt * T_minus[-1]/T_arr[-1]

M = np.array([[M21, M22, M23],
             [M11, M12, M13],
             [M31, M32, M33]])
det_M = np.linalg.det(M)
print(M)
print(det_M)

M11 = 7/2 * heat_flux;      M12 = -chi_hat * dT_plus[0];        M13 = -chi_hat * dT_minus[0]
M21 = 3 * qt;     M22 = -chi_hat * dT_plus[-1] - 1/2 * qt * T_plus[-1]/T_arr[-1];       M23 = -chi_hat * dT_minus[-1] - 1/2 * qt * T_minus[-1]/T_arr[-1]
M31 = qt -  heat_flux;     M32 = -chi_hat * dT_plus[-1] - (-chi_hat * dT_plus[0]);       M33 = -chi_hat * dT_minus[-1] - (-chi_hat * dT_minus[0])

M = np.array([[M11, M12, M13],
              [M21, M22, M23],
             [M31, M32, M33]])
det_M = np.linalg.det(M)
print(M)
print(det_M)

int1 = sigma_hat * V_calc / j0
int2 = 3/2 * 1/result.root * (dT_plus[-1] - dT_plus[0])
int3 = 3/2 * 1/result.root * (dT_minus[-1] - dT_minus[0])
M = np.array([[1, -5/2 * j0, 0, 0],
              [int1, -3/2 * j0 * int1, -3/2 * j0 * int2, -3/2 * j0 * int3],
              [0, 7/2 * heat_flux, -chi_hat * dT_plus[0], -chi_hat * dT_minus[0]],
              [0, 3 * qt, (-chi_hat * dT_plus[-1] - 1/2 * T_plus[-1]/T_arr[-1] * qt), (-chi_hat * dT_minus[-1] - 1/2 * T_minus[-1]/T_arr[-1] * qt)]])
det_M = np.linalg.det(M)
print(M)
print(det_M)
