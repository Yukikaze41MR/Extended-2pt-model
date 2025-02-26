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
def ini_guess(f_t, z_arr, Lz, qu_input, chi): #ode guess
    guess_arr = np.ones((2, z_arr.size))
    guess_arr[0, :] = f_t + qu_input / chi * (Lz - z_arr)
    guess_arr[1, :] = qu_input / (-chi)
    return guess_arr #y_ini = ini_guess(T_t_input, z_arr_input, Lz)

def my_incredible_ode_solver(T_t_input, z_arr_input, initial_guess, qu_input, K_square, loss1, loss2):
    @nb.jit
    def BC(ya, yb):
        return np.array([yb[0] - f_Z, ya[1] - df0_dz])
    @nb.jit
    def ode(z, y):
        f = y[0] # y1 = f
        df_dz = y[1] # y2 = df/dz
        dy2_dz = -K_square * (7/2 * f)**(-3/7) - loss1 * (7/2 * f)**(2/7 * loss2)
        return [df_dz, dy2_dz]
    
    f_Z = 2/7 * T_t_input**(7/2) # f(Z) = f(T_t)
    df0_dz = qu_input / (-chi_hat) # df(Z)/dz = f(q_u)
    solution = solve_bvp(ode, BC, z_arr_input, initial_guess, max_nodes = 1e6, tol = 1e-5)
    f_arr_output = solution.y[0]
    df_arr_output = solution.y[1]
    T_arr_output = (7/2 * solution.y[0])**(2/7)
    z_arr_input = np.linspace(0, Lz, T_arr_output.size)
    dT_arr_output = solution.y[1]/(T_arr_output**(5/2))
    int_loss = simpson(T_arr_output**(beta), x = z_arr_input)
    
    qt_output = np.sqrt(K_square * chi_hat**2 * (T_arr_output[0]**2 - T_arr_output[-1]**2) + qu_input**2 + 2*loss1/(loss2 + 7/2) * (T_arr_output[0]**(loss2 + 7/2) - T_arr_output[-1]**(loss2 + 7/2)))
    voltage_output = (qt_output - qu_input - loss1/chi_hat*int_loss) / np.sqrt(K_square * chi_hat * sigma_hat)#
    
    return T_arr_output, dT_arr_output, voltage_output, qt_output, f_arr_output, df_arr_output

def g(x):
    return my_incredible_ode_solver(Tt, z_space, y_ini, heat_flux, x, alpha, beta)[2] - V_expected 

def ini_guess_p(value, slope, z): #Linear guess
    guess_arr = np.ones((2, z.size))
    guess_arr[0, :] = value + slope * z
    guess_arr[1, :] = slope
    return guess_arr #y_ini = ini_guess(bc1, (bc2-bc1)/Lz, z_arr_input)
    
def perturbation_ode_solver(f0_input, z_arr_input, initial_guess, chi_input, sigma_input, j_0, loss1, loss2, bc1 = 1, bc2 = 0.1):

    f0_interp = interp1d(np.linspace(0, Lz, f0_input.size), f0_input, kind='cubic', bounds_error=False)
    
    def BC_perturbation(ya, yb):
        return np.array([ya[0] - bc1, yb[0] - bc2])
    def ode_perturbation(z, y):
        f1 = y[0] # y1 = f
        df_dz = y[1] # y2 = df/dz
        #f0 = np.interp(np.linspace(0, 1, f1.size), np.linspace(0, 1, f0_input.size), f0_input, left=f0_input[0], right=f0_input[-1])
        f0 = f0_interp(z)
        df2_dz = (-3/2 * j_0**2/sigma_input * (7/2)**(-10/7) * f0**(-10/7) * f1 - loss1/2 * (7/2)**(2/7*loss2) * f0**(2/7*loss2) - loss1 * loss2 * (7/2)**(2/7*loss2 - 1) * f0**(2/7*loss2 - 1) * f1) / -chi_input
        return [df_dz, df2_dz]
    
    solution = solve_bvp(ode_perturbation, BC_perturbation, z_arr_input, initial_guess, max_nodes = 1e6, tol = 1e-3)
    (solution.message != 'The algorithm converged to the desired accuracy.') and print(solution.message)
    return solution.y[0], solution.y[1]

@nb.jit
def calculator_detM(T0_arr, T1_arr, T2_arr, deri1, deri2, int1, int2, int3, int4, qt, qu, voltage, j0, sigma, chi, alpha, beta):
    #int1 = sigma_hat * V_calc / j0
    #int2 = simpson(T_arr**(beta), x = z_space) # int of T_arr**beta
    int_pm1 = 2/3 * chi*sigma/j0**2 * (deri1[-1] - deri1[0]) - alpha/3 * sigma/j0**2 * int2 - 2/3 * alpha * beta * sigma/j0**2 * int3
    int_pm2 = 2/3 * chi*sigma/j0**2 * (deri2[-1] - deri2[0]) - alpha/3 * sigma/j0**2 * int2 - 2/3 * alpha * beta * sigma/j0**2 * int4

    M11 = 2 * j0/sigma * int1
    M12 = -(5 * j0**2/sigma * int1 + alpha * beta * int2) 

    M21 = int1
    M22 = -3/2 * j0 * int1
    M23 = -3/2 * j0 * int_pm1
    M24 = -3/2 * j0 * int_pm2

    M32 = 7/2 * qu
    M33 = -chi * deri1[0]
    M34 = -chi * deri2[0]
    M42 = 3 * qt
    M43 = -chi * deri1[-1] - 1/2 * T1_arr[-1]/T0_arr[-1] * qt
    M44 = -chi * deri2[-1] - 1/2 * T2_arr[-1]/T0_arr[-1] * qt
    Matrix = np.array([[M11, M12, 0, 0],
                    [M21, M22, M23, M24],
                    [0, M32, M33, M34,],
                    [0, M42, M43, M44]])

    return np.linalg.det(Matrix), Matrix

K_sqr_ini = 1e4
Lz = 50.0 # [m]
m = 1.67e-27# [kg] 
gamma_hat = 7.0 * np.sqrt(2/m)
e = 1.60218e-19 # [C]
V_expected = 25 # [V]
Tt = 100 # [eV]
epsilon0 = 8.8541878188e-12
me = 9.1093837015e-31
coulomb_log = 13
chi_hat = 9.6 * ((2 * np.pi)**(3/2) * epsilon0**2) / (me**(1/2) * e**(1/2) * coulomb_log) # [W/m eV^-7/2]
sigma_hat = 0.6125 * chi_hat

alpha = 1 # Loss multiplier
beta = 3/2 # Loss exp

N = int(20)
N_nodes = int(1e4)
qu_arr = np.logspace(7, 9, N)
N = qu_arr.size
z_arr = np.linspace(0, Lz, N_nodes)
y_ini = np.zeros((2, N_nodes))
Tu_arr = np.zeros(N); qt_arr = np.zeros(N); nt_arr = np.zeros(N); nu_arr = np.zeros(N); K_arr = np.zeros(N)
M = np.zeros((qu_arr.size, 4, 4))
det_M = np.zeros(N)
K_arr[-1] = K_sqr_ini

y_ini = ini_guess(Tt, z_arr, Lz, qu_arr[0], chi_hat)
z_space = z_arr
start_time = time.time()

for i in range(N):
    heat_flux = qu_arr[i]
    T_arr = np.zeros((N_nodes)); f_arr = np.zeros((N_nodes)); dT_arr = np.zeros((N_nodes)); df_arr = np.zeros((N_nodes))
    result = root_scalar(g, x0=K_arr[i-1], method='secant') #root_scalar(g, x0=K_sqr_ini, method='secant')
    if math.isnan(result.root) or (result.root==0):
        print('Break error in index = ', i)
        break
    K_arr[i] = result.root
    T_arr, dT_arr, V_calc, qt, f_arr, df_arr = my_incredible_ode_solver(Tt, z_space, y_ini, heat_flux, result.root, alpha, beta)
    y_ini = np.array([f_arr, df_arr])
    z_space = np.linspace(0, Lz, f_arr.size)
    Tu_arr[i] = T_arr[0]
    qt_arr[i] = qt
    nt_arr[i] = qt / (gamma_hat * (e * Tt)**(3/2) * 1e18)
    
    if i==0:
        yp_ini1 = ini_guess_p(T_arr[-1]**(5/2)*1, (T_arr[-1]**(5/2)*1-T_arr[-1]**(5/2)*1)/Lz, z_space)
        yp_ini2 = ini_guess_p(T_arr[-1]**(5/2)*1, (T_arr[-1]**(5/2)*-1-T_arr[-1]**(5/2)*1)/Lz, z_space)
    
    yp_ini1[0] = np.interp(np.linspace(0, 1, f_arr.size), np.linspace(0, 1, yp_ini1[0].size), yp_ini1[0])
    yp_ini1[1] = np.interp(np.linspace(0, 1, f_arr.size), np.linspace(0, 1, yp_ini1[1].size), yp_ini1[1])
    yp_ini2[0] = np.interp(np.linspace(0, 1, f_arr.size), np.linspace(0, 1, yp_ini2[0].size), yp_ini2[0])
    yp_ini2[1] = np.interp(np.linspace(0, 1, f_arr.size), np.linspace(0, 1, yp_ini2[1].size), yp_ini2[1])

    j0 = np.sqrt(result.root * chi_hat * sigma_hat)
    T_plus, dT_plus = perturbation_ode_solver(f_arr, z_space, yp_ini1, chi_hat, sigma_hat, j0, alpha, beta, T_arr[0]**(5/2)*1, T_arr[-1]**(5/2)*1)
    T_minus, dT_minus = perturbation_ode_solver(f_arr, z_space, yp_ini2, chi_hat, sigma_hat, j0, alpha, beta, T_arr[0]**(5/2)*1, T_arr[-1]**(5/2)*-1)

    yp_ini1 = np.array([T_plus, dT_plus])
    yp_ini2 = np.array([T_minus, dT_minus])
    T_plus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, T_plus.size), T_plus)
    T_minus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, T_minus.size), T_minus)
    dT_plus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, dT_plus.size), dT_plus)
    dT_minus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, dT_minus.size), dT_minus)

    T_plus = T_plus / T_arr**(5/2)
    T_minus = T_minus / T_arr**(5/2)

    int1 = sigma_hat * V_calc / j0
    int2 = simpson(T_arr**(beta), x = z_space) # int of T_arr**beta
    int3 = simpson(T_arr**(beta-1) * T_plus, x = z_space)
    int4 = simpson(T_arr**(beta-1) * T_minus, x = z_space)
    
    #(T0_arr, T1_arr, T2_arr, deri1, deri2, int1, int2, int3, int4, qt, qu, voltage, j0, sigma, chi, alpha, beta)
    det_M[i], M[i] = calculator_detM(T_arr, T_plus, T_minus, dT_plus, dT_minus, int1, int2, int3, int4, qt, heat_flux, V_calc, j0, sigma_hat, chi_hat, alpha, beta)
    
    result.clear()

end_time = time.time()
print(f'Loop time = {(end_time - start_time):.6f} s')
nu_arr = 2*nt_arr*Tt/Tu_arr

print('Have negative value = ', np.in1d(-1, np.sign(det_M)))
plt.figure()
plt.title(f'Voltage = {V_expected} V, Alpha = {alpha}, Beta = {beta}', fontsize = 13)
plt.axhline(0, color='black', linestyle='--')
plt.semilogx(qu_arr, det_M, color = 'b')
plt.semilogx(qu_arr[np.nonzero(det_M>0)], det_M[np.nonzero(det_M>0)], color = 'r')
plt.xlabel('$q_u$')
#plt.legend(fontsize = 10)
plt.xlim(qu_arr[0], qu_arr[-1])
plt.show()