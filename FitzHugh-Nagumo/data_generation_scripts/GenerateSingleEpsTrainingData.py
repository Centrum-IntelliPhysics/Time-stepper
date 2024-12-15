import sys
sys.path.append('../')

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from EulerTimestepper import fhn_euler_timestepper

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

def sampleSinusoidalInitial():
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

    base_fn = lambda x: np.sin(2*np.pi*x - np.pi/2.0) + 1
    x_array = np.linspace(0.0, 1.0, N)
    initial_u_means = []
    initial_v_means = []
    store_directory = './../data/singleparameter/'
    rng = rd.RandomState()
    for j in range(n_initials):
        print('initial ', j)
        u = base_fn(x_array) * rng.normal(0.0, 1.0, N) - 0.25
        v = base_fn(x_array) * rng.normal(0.0, 1.0, N)
        
        #plt.plot(x_array, u)
        #plt.plot(x_array, v)
        #plt.show()
        initial_u_means.append(np.average(u))
        initial_v_means.append(np.average(v))
        evolution = timeSimulation(u, v, dx, dt, T, dT, params)

        # Store the time evolution
        np.save(store_directory + 'FHN_SingleEpsilon_SineEvolution_Initial=' + str(j) + '_eps=' + str(eps).replace('.', 'p') + '_dT=' + str(dt).replace('.', 'p') + '.npy', evolution)

    # Make a histogram plot
    plt.hist(initial_u_means, bins=50, density=True, label=r'Histogram of $<u>$')
    plt.legend()
    plt.figure()
    plt.hist(initial_v_means, bins=50, density=True, label=r'Histogram of $<v>$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    sampleSinusoidalInitial()
