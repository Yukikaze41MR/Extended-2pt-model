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
@nb.jit
def ini_guess(f_t, z_arr, Lz, qu_input, chi): #ode guess
    guess_arr = np.ones((2, z_arr.size))
    guess_arr[0, :] = f_t + qu_input / chi * (Lz - z_arr)
    guess_arr[1, :] = qu_input / (-chi)
    return guess_arr #y_ini = ini_guess(T_t_input, z_arr_input, Lz)

def my_incredible_ode_solver(T_t_input, z_arr_input, initial_guess, qu_input, K_square):
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
    solution = solve_bvp(ode, BC, z_arr_input, initial_guess, max_nodes = 1e6, tol = 1e-5)
    f_arr_output = solution.y[0]
    df_arr_output = solution.y[1]
    T_arr_output = (7/2 * solution.y[0])**(2/7)
    dT_arr_output = solution.y[1]/(T_arr_output**(5/2))
    qt_output = np.sqrt(K_square * chi_hat**2 * (T_arr_output[0]**2 - T_arr_output[-1]**2) + qu_input**2)
    voltage_output = (qt_output-qu_input) / np.sqrt(K_square * chi_hat * sigma_hat)
    
    return T_arr_output, dT_arr_output, voltage_output, qt_output, f_arr_output, df_arr_output

def g(x):
    return my_incredible_ode_solver(Tt, z_space, y_ini, heat_flux, x)[2] - V_expected 

@nb.jit
def ini_guess_p(value, slope, z): #Linear guess
    guess_arr = np.ones((2, z.size))
    guess_arr[0, :] = value + slope * z
    guess_arr[1, :] = slope
    return guess_arr #y_ini = ini_guess(bc1, (bc2-bc1)/Lz, z_arr_input)
    
def perturbation_ode_solver(f0_input, z_arr_input, initial_guess, Ksqr_input, bc1 = 1, bc2 = 1):
    
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
    solution = solve_bvp(ode_perturbation, BC_perturbation, z_arr_input, initial_guess, max_nodes = 1e6, tol = 1e-3)
    (solution.message != 'The algorithm converged to the desired accuracy.') and print(solution.message)
    return solution.y[0], solution.y[1]

@nb.jit
def calculator_detM(T0, T_plus, T_minus, dT_plus, dT_minus, qu, qt, chi_input):
    M11 = 7/2 * qu;      M12 = -chi_hat * dT_plus[0];        M13 = -chi_input * dT_minus[0]
    M21 = qt + 5/2 * qu;     M22 = -chi_hat * dT_plus[-1];       M23 = -chi_input * dT_minus[-1]
    M31 = 2 * qt - 5/2 * qu;     M32 = - 1/2 * qt * T_plus[-1]/T0[-1];        M33 = - 1/2 * qt * T_minus[-1]/T0[-1]

    Matrix = np.array([[M21, M22, M23],
                       [M11, M12, M13],
                       [M31, M32, M33]])
    return np.linalg.det(Matrix), Matrix

Lz = 50.0 # [m]
m = 1.67e-27# [kg] 
gamma_hat = 7.0 * np.sqrt(2/m)
e = 1.60218e-19 # [C]
epsilon0 = 8.8541878188e-12
me = 9.1093837015e-31
coulomb_log = 13
chi_hat = 9.6 * ((2 * np.pi)**(3/2) * epsilon0**2) / (me**(1/2) * e**(1/2) * coulomb_log) # [W/m eV^-7/2]
sigma_hat = 0.6125 * chi_hat
Tt = 75 # [eV]
print('T_t =', Tt,'[eV]')

N = int(80)
N_nodes = int(1e4)
z_arr = np.linspace(0, Lz, N_nodes)
#qu_arr = np.concatenate([np.arange(1e6, 1.1e7, 0.5e6), np.arange(1e7, 1.1e8, 0.5e7), np.arange(1e8, 1.1e9, 0.5e8), np.arange(1e9, 1.1e10, 0.5e9)])
#qu_arr = np.concatenate([np.arange(1e5, 1.1e6, 0.5e5), np.arange(1e6, 1.1e7, 0.5e6), np.arange(1e7, 1.1e8, 0.5e7), np.arange(1e8, 1.1e9, 0.5e8)]) #zero arr

#qu_arr = np.concatenate([np.logspace(6, 7, 20)[:-1], np.logspace(7, 8, 20)[:-1], np.logspace(8, 9, 20)[:-1], np.logspace(9, 10, 20)])
#qu_arr = np.concatenate([np.logspace(5, 6, 20)[:-1], np.logspace(6, 7, 20)[:-1], np.logspace(7, 8, 20)[:-1], np.logspace(8, 9, 20)]) #Ksi arr = 40, zero arr = 20
qu_arr = np.concatenate([np.logspace(5, 6, 40)[:-1], np.logspace(6, 7, 40)[:-1], np.logspace(7, 8, 40)[:-1], np.logspace(8, 9, 40)]) #Ksi arr = 40, zero arr = 20
N = qu_arr.size
#qu_arr = np.logspace(6, 10, N)

#voltage_arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]) #100eV
voltage_arr = np.array([125, 137.5, 150, 160, 162.5, 165, 166, 166.5]) #Low Tt 75
#voltage_arr = np.array([2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25]) #High Tt 500
#voltage_arr = np.array([2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 35, 40, 45, 50]) #High Tt 300

#voltage_arr = np.array([125, 150, 155, 157.5, 160, 162.5, 165, 167.5, 170]) #Test_arr
Nv = voltage_arr.size

x_parameter = np.zeros((Nv)); y_parameter = np.zeros((Nv)); z_parameter = np.zeros((Nv)); w_parameter = np.zeros((Nv))
instability_x = np.zeros((Nv)); instability_y = np.zeros((Nv))
stability_x = np.zeros((Nv)); stability_y = np.zeros((Nv))
stability_pack = np.zeros((4, Nv))

i_start = 0
K_sqr_ini = 5e6
K_sqr = K_sqr_ini
print('Ksqr = ', K_sqr)
print('Initial calculated voltage =', my_incredible_ode_solver(Tt, z_arr, ini_guess(Tt, z_arr, Lz, qu_arr[0], chi_hat), qu_arr[0], K_sqr_ini)[2], '[V]')

for v in range(Nv):
    V_expected = voltage_arr[v]

    z_space = z_arr
    qt_arr = np.zeros(N); nt_arr = np.zeros(N); Tu_arr = np.zeros(N);
    p1 = np.zeros(N); p2 = np.zeros(N); p3 = np.zeros(N); p4 = np.zeros(N);
    M = np.zeros((N, 3, 3))
    det_M = np.zeros(N)

    y_ini = np.zeros((2, N_nodes))
    y_ini = ini_guess(Tt, z_arr, Lz, qu_arr[0], chi_hat)
    start_time = time.time()
    for i in range(i_start, N):
        heat_flux = qu_arr[i]
        T_arr = np.zeros((N_nodes)); f_arr = np.zeros((N_nodes)); dT_arr = np.zeros((N_nodes)); df_arr = np.zeros((N_nodes))
        result = root_scalar(g, x0=K_sqr, method='secant', rtol = 1e-5) #root_scalar(g, x0=K_sqr_ini, method='secant')
        if math.isnan(result.root) or (result.root==0):
            print('Break error in index = ', i)
            break
        K_sqr = result.root
        T_arr, dT_arr, V_calc, qt, f_arr, df_arr = my_incredible_ode_solver(Tt, z_space, y_ini, heat_flux, result.root)
        y_ini = np.array([f_arr, df_arr])
        z_space = np.linspace(0, Lz, f_arr.size)
        Tu_arr[i] = T_arr[0]
        qt_arr[i] = qt
        nt_arr[i] = qt / (gamma_hat * (e * Tt)**(3/2))
        p1[i] = (nt_arr[i] * Tt) / heat_flux**(6/7)
        p2[i] = V_calc / heat_flux**(2/7)
        p3[i] = V_calc/Tt
        p4[i] = heat_flux/Tt**(7/2)
        
        if i==0:
            yp_ini1 = ini_guess_p(T_arr[-1]**(5/2)*1, (T_arr[-1]**(5/2)*1-T_arr[-1]**(5/2)*1)/Lz, z_space)
            yp_ini2 = ini_guess_p(T_arr[-1]**(5/2)*1, (T_arr[-1]**(5/2)*-1-T_arr[-1]**(5/2)*1)/Lz, z_space)
        
        yp_ini1 = [np.interp(np.linspace(0, 1, f_arr.size), np.linspace(0, 1, yp_ini1[0].size), yp_ini1[0]), np.interp(np.linspace(0, 1, f_arr.size), np.linspace(0, 1, yp_ini1[1].size), yp_ini1[1])]
        yp_ini2 = [np.interp(np.linspace(0, 1, f_arr.size), np.linspace(0, 1, yp_ini2[0].size), yp_ini2[0]), np.interp(np.linspace(0, 1, f_arr.size), np.linspace(0, 1, yp_ini2[1].size), yp_ini2[1])]

        j0 = np.sqrt(result.root * chi_hat * sigma_hat)
        T_plus, dT_plus = perturbation_ode_solver(f_arr, z_space, yp_ini1, K_sqr, T_arr[0]**(5/2)*1, T_arr[-1]**(5/2)*1)
        T_minus, dT_minus = perturbation_ode_solver(f_arr, z_space, yp_ini2, K_sqr, T_arr[0]**(5/2)*1, T_arr[-1]**(5/2)*-1)
    
        yp_ini1 = np.array([T_plus, dT_plus])
        yp_ini2 = np.array([T_minus, dT_minus])
        T_plus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, T_plus.size), T_plus)
        T_minus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, T_minus.size), T_minus)
        dT_plus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, dT_plus.size), dT_plus)
        dT_minus = np.interp(np.linspace(0, 1, T_arr.size), np.linspace(0, 1, dT_minus.size), dT_minus)

        T_plus = T_plus / T_arr**(5/2)
        T_minus = T_minus / T_arr**(5/2)
        #detcalc3(T_arr, T_plus, T_minus, dT_plus, dT_minus, heat_flux, qt, chi_hat)
        det_M[i], M[i] = calculator_detM(T_arr, T_plus, T_minus, dT_plus, dT_minus, heat_flux, qt, chi_hat)
        
        if det_M[i_start] > 0:
            print('Stable or Solver crashed')
            break
        elif (det_M[i-1] < 0) and (det_M[i] > 0):
            result.clear()
            break

        result.clear()
    end_time = time.time()
    print('Ksqr = ', K_sqr)
    print(f'Loop time for {V_expected}V = {(end_time - start_time):.6f} s')
    #print(np.nonzero(det_M<0)[0][-1], np.nonzero(det_M>0)[0][0])
    stability_pack[0, v] = V_expected #voltage
    stability_pack[1, v] = (nt_arr[np.nonzero(det_M<0)[0][-1]] + nt_arr[np.nonzero(det_M>0)[0][0]]) * Tt /2 #pressure
    stability_pack[2, v] = (qu_arr[np.nonzero(det_M<0)[0][-1]] + qu_arr[np.nonzero(det_M>0)[0][0]])/2 #heatflux
    Tu = ((Tu_arr[np.nonzero(det_M<0)[0][-1]] + Tu_arr[np.nonzero(det_M>0)[0][0]])/2)
    stability_pack[3, v] = Tt**(7/2) / (Tu**(7/2) - Tt**(7/2)) #ksi

    i_start = np.maximum(np.nonzero(det_M<0)[0][-1]-20, 0)
    x_parameter[v] = (p2[np.nonzero(det_M<0)[0][-1]] + p2[np.nonzero(det_M>0)[0][0]])/2
    y_parameter[v] = (p1[np.nonzero(det_M<0)[0][-1]] + p1[np.nonzero(det_M>0)[0][0]])/2

    z_parameter[v] = (p3[np.nonzero(det_M<0)[0][-1]] + p3[np.nonzero(det_M>0)[0][0]])/2
    w_parameter[v] = (p4[np.nonzero(det_M<0)[0][-1]] + p4[np.nonzero(det_M>0)[0][0]])/2
    

    instability_x[v] = p3[np.nonzero(det_M<0)[0][-1]] #Used for verify stability/instability side
    instability_y[v] = p4[np.nonzero(det_M<0)[0][-1]]
    stability_x[v] = p3[np.nonzero(det_M>0)[0][0]]
    stability_y[v] = p4[np.nonzero(det_M>0)[0][0]]
