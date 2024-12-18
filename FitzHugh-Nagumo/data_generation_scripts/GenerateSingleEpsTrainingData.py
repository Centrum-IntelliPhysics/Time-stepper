import sys
sys.path.append('../')

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from EulerTimestepper import fhn_euler_timestepper

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

def sampleSigmoidInitial():
    eps = 0.1
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    T = 20.0
    N = 200
    M = 2*N
    L = 20.0
    dt = 1.e-3
    dT = 10 * dt
    dx = L / N
    n_initials = 1000
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    rng = rd.RandomState(seed=100)
    x_array = np.linspace(0.0, L, N)
    u_x_shifts = rng.uniform(4.0, 16.0, size=n_initials)
    u_x_scale = rng.uniform(0.5, 2, size=n_initials)
    v_x_shifts = rng.uniform(5.0, 15.0, size=n_initials)
    v_x_scale = rng.uniform(0.5, 2, size=n_initials)
    y_shifts = rng.uniform(-2.0, 0.0, size=n_initials)
    y_scales = rng.uniform(1.0, 4.0, size=n_initials)
    initial_u_means = []
    initial_v_means = []
    store_directory = './../data/singleparameter/'
    rng = rd.RandomState()
    for j in range(n_initials):
        print('initial ', j)
        u = sigmoid(x_array, u_x_shifts[j], y_shifts[j], u_x_scale[j], y_scales[j])
        v = sigmoid(x_array, v_x_shifts[j], y_shifts[j], v_x_scale[j], y_scales[j])
        
        initial_u_means.append(np.average(u))
        initial_v_means.append(np.average(v))
        evolution = timeSimulation(u, v, dx, dt, T, dT, params)

        # Store the time evolution
        np.save(store_directory + 'FHN_SingleEpsilon_SigmoidEvolution_Initial=' + str(j) + '_eps=' + str(eps).replace('.', 'p') + '_dT=' + str(dt).replace('.', 'p') + '.npy', evolution)

    # Make a histogram plot
    plt.hist(initial_u_means, bins=50, density=True, label=r'Histogram of $<u>$')
    plt.legend()
    plt.figure()
    plt.hist(initial_v_means, bins=50, density=True, label=r'Histogram of $<v>$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    sampleSigmoidInitial()
