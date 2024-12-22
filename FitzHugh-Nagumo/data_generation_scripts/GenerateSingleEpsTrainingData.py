import sys
sys.path.append('../')

import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import matplotlib.pyplot as plt

from EulerTimestepper import fhn_euler_timestepper, calculateSteadyState, psi

# Original sigmoid between 0 and 1. To make it between -1 and 1, shift by y_center=-0.5 and y_scale=2
def sigmoid(x, x_center=0.0, y_center=0.0, x_scale=1.0, y_scale=1.0):
    return y_scale / (1.0 + np.exp(-(x  - x_center)/x_scale)) + y_center

def timeSimulation(u0, v0, dx, dt, T, dT, params):
    n_timesteps = int(T / dT)
    solution_slices = np.zeros((n_timesteps+1, len(u0) + len(v0)))
    solution_slices[0,:] = np.concatenate((u0, v0))

    u = np.copy(u0)
    v = np.copy(v0)
    for i in range(n_timesteps):
        u, v = fhn_euler_timestepper(u, v, dx, dt, dT, params, verbose=False)
        solution_slices[i+1,:] = np.concatenate((u, v))
    return solution_slices

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
    dT = 100 * dt
    dx = L / N

    # Calculate the steady state by timestepping, followed by Newton-Krylov
    T_psi = 1.0
    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 10, 0.0, 2.0, 0.1)
    u0, v0 = fhn_euler_timestepper(u0, v0, dx, dt, 100.0, params, verbose=False)
    x0 = np.concatenate((u0, v0))
    x_ss = calculateSteadyState(x0, T_psi, dx, dt, params)
    u_ss = x_ss[0:N]
    v_ss = x_ss[N:]
    print('Final psi :', lg.norm(psi(x_ss, T_psi, dx, dt, params)))
    
    rng = rd.RandomState(seed=100)
    min_n = 0
    max_n = N // 2
    max_delta = 0.1
    n_initials = 1000
    for j in range(n_initials):
        print('Initial #', j+1)
        delta_u = max_delta * rng.uniform(-1.0, 1.0)
        delta_v = max_delta * rng.uniform(-1.0, 1.0)
        n_u = rng.randint(min_n, max_n+1)
        n_v = rng.randint(min_n, max_n+1)
        u = u_ss + delta_u * np.cos(2.0 * np.pi * n_u * x_array / L)
        v = v_ss + delta_v * np.cos(2.0 * np.pi * n_v * x_array / L)
        evolution = timeSimulation(u, v, dx, dt, T, dT, params)

        # Store the time evolution
        np.save('./../data/singleparameter/FHN_SingleEpsilon_SinePerturbationEvolution_Initial=' + str(j) + '_eps=' + str(eps).replace('.', 'p') + '_dT=' + str(dt).replace('.', 'p') + '.npy', evolution)

def plotTransients():
    n_initials = 1000
    for j in range(n_initials):
        evolution = np.load('./../data/singleparameter/FHN_SingleEpsilon_SinePerturbationEvolution_Initial=' + str(j) + '_eps=0p1_dT=0p001.npy')
        norm_evolution = lg.norm(evolution, axis=1)
        plt.plot(norm_evolution, color='tab:blue')
    plt.xlabel(r'$t$ [s/10]')
    plt.ylabel('Transient Norms')
    plt.legend([r'$\sqrt{\|u(t)\|^2 + \|v(t)\|^2}$'])
    plt.show()

if __name__ == '__main__':
    sampleCosinePerturbations()
    plotTransients()