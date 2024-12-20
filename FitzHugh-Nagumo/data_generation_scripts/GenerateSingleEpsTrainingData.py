import sys
sys.path.append('../')

import numpy as np
import numpy.linalg as lg
import numpy.random as rd

from EulerTimestepper import fhn_euler_timestepper, calculateSteadyState, psi

# Original sigmoid between 0 and 1. To make it between -1 and 1, shift by y_center=-0.5 and y_scale=2
def sigmoid(x, x_center=0.0, y_center=0.0, x_scale=1.0, y_scale=1.0):
    return y_scale / (1.0 + np.exp(-(x  - x_center)/x_scale)) + y_center

def timeSimulation(u0, v0, dx, dt, T, dT, params):
    n_entries = int(T / dT) + 1
    solution_slices = np.zeros((n_entries, len(u0) + len(v0)))
    solution_slices[0,:] = np.concatenate((u0, v0))

    u = np.copy(u0)
    v = np.copy(v0)
    report_n = int(dT / dt)
    for i in range(n_entries-1):
        u, v = fhn_euler_timestepper(u, v, dx, dt, dT, params, verbose=False)

        if i % report_n == 0:
            index = i // report_n
            solution_slices[index+1,:] = np.concatenate((u, v))

    return solution_slices

def sampleNoiseInitial():
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    T = 2.0
    N = 200
    M = 2*N
    L = 20.0
    dt = 1.e-3
    dx = L / N
    n_initials = 1000

    # Load the bifurcation diagram to determine a good initial point
    eps = 0.1
    load_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
    bf_data = np.load(load_directory + 'euler_bf_diagram.npy')
    x0 = bf_data[0,0:M]
    u0 = x0[0:N]
    v0 = x0[N:]

    store_directory = './../data/singleparameter/'
    rng = rd.RandomState()
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}
    for j in range(n_initials):
        print('initial ', j)
        u = u0 + 0.01*rng.normal(0.0, 1.0, N)
        v = v0 + 0.01*rng.normal(0.0, 1.0, N)
        evolution = timeSimulation(u, v, dx, dt, T, dt, params)

        # Store the time evolution
        np.save(store_directory + 'FHN_SingleEpsilon_Evolution_Initial=' + str(j) + '_eps=' + str(eps).replace('.', 'p') + '_dT=' + str(dt).replace('.', 'p') + '.npy', evolution)

def sampleCosinePerturbations():
    # Model parameters
    eps = 0.1
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Simulation parameters
    T = 100.0
    N = 200
    L = 20.0
    dt = 1.e-3
    dT = 10 * dt
    dx = L / N

    # Calculate the steady state by timestepping, followed by Newton-Krylov
    T_psi = 1.0
    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 10, 0.0, 2.0, 0.1)
    x0 = np.concatenate((u0, v0))
    u0, v0 = fhn_euler_timestepper(u0, v0, dx, dt, 100.0, params, verbose=False)
    x0 = np.concatenate((u0, v0))
    x_ss = calculateSteadyState(x0, T_psi, dx, dt, params)
    u_ss = x_ss[0:N]
    v_ss = x_ss[N:]
    print('Final psi :', lg.norm(psi(x_ss, T_psi, dx, dt, params)))
    
    rng = rd.RandomState(seed=100)
    min_n = 0
    max_n = N // 2
    max_delta = 0.01
    n_initials = 1000
    for j in range(n_initials):
        print('initial ', j)
        delta_u = max_delta * rng.uniform(-1.0, 1.0)
        delta_v = max_delta * rng.uniform(-1.0, 1.0)
        n_u = rng.randint(min_n, max_n+1)
        n_v = rng.randint(min_n, max_n+1)
        u = u_ss + delta_u * np.cos(2.0 * np.pi * n_u * x_array / L)
        v = v_ss + delta_v * np.cos(2.0 * np.pi * n_v * x_array / L)
        evolution = timeSimulation(u, v, dx, dt, T, dT, params)

        # Store the time evolution
        np.save('./../data/singleparameter/FHN_SingleEpsilon_SinePerturbationEvolution_Initial=' + str(j) + '_eps=' + str(eps).replace('.', 'p') + '_dT=' + str(dt).replace('.', 'p') + '.npy', evolution)

if __name__ == '__main__':
    sampleCosinePerturbations()
