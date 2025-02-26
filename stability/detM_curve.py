import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar
import time
import warnings
import math
from scipy.integrate import simpson
from scipy.interpolate import interp1d

#warnings.filterwarnings("ignore", category=RuntimeWarning)

def my_incredible_ode_solver(T_t_input, z_arr_input, qu_input, K_square):
    def BC(ya, yb):
        return np.array([yb[0] - f_Z, ya[1] - df0_dz])
    def ode(z, y):
        f = y[0] # y1 = f
        df_dz = y[1] # y2 = df/dz
        dy2_dz = -K_square * (7/2 * f)**(-3/7)
        return [df_dz, dy2_dz]
    def ini_guess(f_t, z_arr, Lz): #Linear guess
        guess_arr = np.ones((2, z_arr.size))
        guess_arr[0, :] = f_t + qu_input / chi_hat * (Lz - z_arr)
        guess_arr[1, :] = qu_input / (-chi_hat)
        return guess_arr

    f_Z = 2/7 * T_t_input**(7/2) # f(Z) = f(T_t)
    df0_dz = qu_input / (-chi_hat) # df(Z)/dz = f(q_u)
    y_ini = ini_guess(T_t_input, z_arr_input, Lz)
    solution = solve_bvp(ode, BC, z_arr_input, y_ini, max_nodes = 1e6, tol = 1e-5)
    f_arr_output = solution.y[0]
    T_arr_output = (7/2 * solution.y[0])**(2/7)
    dT_arr_output = solution.y[1]/(T_arr_output**(5/2))
    
    qt_output = np.sqrt(K_square * chi_hat**2 * (T_arr_output[0]**2 - T_arr_output[-1]**2) + qu_input**2)
    voltage_output = (qt_output-qu_input) / np.sqrt(K_square * chi_hat * sigma_hat)
    
    return T_arr_output, dT_arr_output, voltage_output, qt_output, f_arr_output

def g(x):
    return my_incredible_ode_solver(Tt, z_arr, heat_flux, x)[2] - V_expected

def perturbation_ode_solver(f0_input, z_arr_input, Ksqr_input, bc1 = 1, bc2 = 1):
    
    f0_interp = interp1d(np.linspace(0, Lz, f0_input.size), f0_input, kind='cubic', bounds_error=False)

    def BC_perturbation(ya, yb):
        return np.array([ya[0] - bc1, yb[0] - bc2])
    def ode_perturbation(z, y):
        f1 = y[0] # y1 = f
        df_dz = y[1] # y2 = df/dz
        #f0 = np.interp(np.linspace(0, 1, f1.size), np.linspace(0, 1, f0_input.size), f0_input, left=f0_input[0], right=f0_input[-1])
        f0 = f0_interp(z)
        df2_dz = 3/2 * Ksqr_input * (7/2)**(-10/7) * f0**(-10/7) * f1
        return [df_dz, df2_dz]
    def ini_guess(value1, slope, z): #Linear guess
        guess_arr = np.ones((2, z.size))
        guess_arr[0] = value1 + slope * z
        guess_arr[1] = slope
        return guess_arr
    
    y_ini = ini_guess(bc1, (bc2-bc1)/Lz, z_arr_input)
    solution = solve_bvp(ode_perturbation, BC_perturbation, z_arr_input, y_ini, max_nodes = 1e6, tol = 1e-3)
    (solution.message != 'The algorithm converged to the desired accuracy.') and print(solution.message)
    return solution.y[0], solution.y[1]

@nb.jit
def detcalc1(T0, T_plus, T_minus, dT_plus, dT_minus, qu, qt, Ksqr, chi_input, sigma_input, voltage, j):
    int1 = sigma_input * voltage / j
    int2 = 2/3 * 1/Ksqr * (dT_plus[-1] - dT_plus[0])
    int3 = 2/3 * 1/Ksqr * (dT_minus[-1] - dT_minus[0])
    Matrix = np.array([[1, -5/2 * j, 0, 0],
                [int1, -3/2 * j * int1, -3/2 * j * int2, -3/2 * j * int3],
                [0, 7/2 * qu, -chi_input * dT_plus[0], -chi_input * dT_minus[0]],
                [0, 3 * qt, (-chi_input * dT_plus[-1] - 1/2 * T_plus[-1]/T0[-1] * qt), (-chi_input * dT_minus[-1] - 1/2 * T_minus[-1]/T0[-1] * qt)]])
    return np.linalg.det(Matrix), Matrix

@nb.jit
def detcalc2(T0, T_plus, T_minus, dT_plus, dT_minus, qu, qt, chi_input):
    M11 = 7/2 * qu
    M12 = -chi_input * dT_plus[0]
    M13 = -chi_input * dT_minus[0]
    M21 = 3 * qt
    M22 = -chi_input * dT_plus[-1] - 1/2 * qt * T_plus[-1]/T0[-1]
    M23 = -chi_input * dT_minus[-1] - 1/2 * qt * T_minus[-1]/T0[-1]
    M31 = qt - qu
    M32 = -chi_input * dT_plus[-1] - (-chi_input * dT_plus[0])
    M33 = -chi_input * dT_minus[-1] - (-chi_input * dT_minus[0])
    
    Matrix = np.array([[M11, M12, M13],
                [M21, M22, M23],
                [M31, M32, M33]])
    return np.linalg.det(Matrix), Matrix

@nb.jit
def detcalc2_mk2(T0, T_plus, T_minus, dT_plus, dT_minus, qu, qt, chi_input):
    M11 = 7/2 * qu
    M12 = -chi_input * dT_plus[0]
    M13 = -chi_input * dT_minus[0]
    M21 = 3 * qt
    M22 = -chi_input * dT_plus[-1] - 1/2 * qt * T_plus[-1]/T0[-1]
    M23 = -chi_input * dT_minus[-1] - 1/2 * qt * T_minus[-1]/T0[-1]
    M31 = qt - qu + qt
    M32 = -chi_input * dT_plus[-1] - (-chi_input * dT_plus[0])
    M33 = -chi_input * dT_minus[-1] - (-chi_input * dT_minus[0])

    Matrix = np.array([[M11, M12, M13],
                [M21, M22, M23],
                [M31, M32, M33]])
    return np.linalg.det(Matrix), Matrix

@nb.jit
def detcalc3(T0, T_plus, T_minus, dT_plus, dT_minus, qu, qt, chi_input):
    M11 = 7/2 * qu;      M12 = -chi_hat * dT_plus[0];        M13 = -chi_input * dT_minus[0]
    M21 = qt + 5/2 * qu;     M22 = -chi_hat * dT_plus[-1];       M23 = -chi_input * dT_minus[-1]
    M31 = 2 * qt - 5/2 * qu;     M32 = - 1/2 * qt * T_plus[-1]/T0[-1];        M33 = - 1/2 * qt * T_minus[-1]/T0[-1]

    Matrix = np.array([[M21, M22, M23],
                       [M11, M12, M13],
                       [M31, M32, M33]])
    return np.linalg.det(Matrix), Matrix

K_sqr_ini = 1e3
Lz = 50.0 # [m]
m = 1.67e-27# [kg] 
gamma_hat = 7.0 * np.sqrt(2/m)
e = 1.60218e-19 # [C]
V_expected = 150 # [V]
Tt = 100 # [eV]

epsilon0 = 8.8541878188e-12
me = 9.1093837015e-31
coulomb_log = 13
chi_hat = 9.6 * ((2 * np.pi)**(3/2) * epsilon0**2) / (me**(1/2) * e**(1/2) * coulomb_log) # [W/m eV^-7/2]
sigma_hat = 0.6125 * chi_hat

N = int(40)
N_nodes = int(1e4)
qu_arr = np.logspace(7, 9, N)
N = qu_arr.size
z_arr = np.linspace(0, Lz, N_nodes)
Tu_arr = np.zeros(N); qt_arr = np.zeros(N); nt_arr = np.zeros(N); nu_arr = np.zeros(N); K_arr = np.zeros(N)
det_M2 = np.zeros(N)
M2 = np.zeros((qu_arr.size, 3, 3))
det_M3 = np.zeros(N)
M3 = np.zeros((qu_arr.size, 3, 3))
det_M1 = np.zeros(N)
M1 = np.zeros((qu_arr.size, 4, 4))

det_M = np.zeros(N)
M = np.zeros((qu_arr.size, 3, 3))
K_arr[-1] = K_sqr_ini
start_time = time.time()

for i in range(N):
    heat_flux = qu_arr[i]
    T_arr = np.zeros((N_nodes)); f_arr = np.zeros((N_nodes)); dT_arr = np.zeros((N_nodes))
    result = root_scalar(g, x0=K_arr[i-1], method='secant') #root_scalar(g, x0=K_sqr_ini, method='secant')
    if math.isnan(result.root) or (result.root==0):
        print('Break error in index = ', i)
        break
    K_arr[i] = result.root
    T_arr, dT_arr, V_calc, qt, f_arr = my_incredible_ode_solver(Tt, z_arr, heat_flux, result.root)
    z_space = np.linspace(0, Lz, T_arr.size)
    Tu_arr[i] = T_arr[0]
    qt_arr[i] = qt
    nt_arr[i] = qt / (gamma_hat * (e * Tt)**(3/2))
    
    j0 = np.sqrt(result.root * chi_hat * sigma_hat)
    T_plus, dT_plus = perturbation_ode_solver(f_arr, z_space, result.root, 1 * T_arr[0]**(5/2), 1 * T_arr[-1]**(5/2))
    T_minus, dT_minus = perturbation_ode_solver(f_arr, z_space, result.root, 1 * T_arr[0]**(5/2), -1 * T_arr[-1]**(5/2))
    
    T_plus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, T_plus.size), T_plus)
    T_minus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, T_minus.size), T_minus)
    dT_plus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, dT_plus.size), dT_plus)
    dT_minus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, dT_minus.size), dT_minus)

    T_plus = T_plus / T_arr**(5/2)
    T_minus = T_minus / T_arr**(5/2)
    
    det_M[i], M[i] = detcalc2_mk2(T_arr, T_plus, T_minus, dT_plus, dT_minus, heat_flux, qt, chi_hat)
    det_M3[i], M3[i] = detcalc3(T_arr, T_plus, T_minus, dT_plus, dT_minus, heat_flux, qt, chi_hat)
    #det_M2[i], M2[i] = detcalc2(T_arr, T_plus, T_minus, dT_plus, dT_minus, heat_flux, qt, chi_hat)
    #det_M1[i], M1[i] = detcalc1(T_arr, T_plus, T_minus, dT_plus, dT_minus, heat_flux, qt, result.root, chi_hat, sigma_hat, V_calc, j0)
    result.clear()

end_time = time.time()
print(f'Loop time = {(end_time - start_time):.6f} s')
nu_arr = 2*nt_arr*Tt/Tu_arr

print('Have negative value =', np.in1d(-1, np.sign(det_M)))
plt.figure()
plt.title(f'Voltage = {np.round(V_calc, decimals = 2)} V', fontsize = 13)
plt.axhline(0, color='black', linestyle='--')
plt.semilogx(qu_arr[np.nonzero(det_M<0)], det_M[np.nonzero(det_M<0)], color = 'b')
plt.semilogx(qu_arr[np.nonzero(det_M>0)], det_M[np.nonzero(det_M>0)], color = 'r')
plt.xlabel('$q_u$')
#plt.legend(fontsize = 10)
plt.xlim(qu_arr[0], qu_arr[-1])
plt.show()