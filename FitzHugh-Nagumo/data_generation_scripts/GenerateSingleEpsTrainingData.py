import sys
sys.path.append('../')

import numpy as np
import numpy.random as rd

from EulerTimestepper import fhn_euler_timestepper

def timeSimulation(u0, v0, dx, dt, T, dT, params):
    n_entries = int(T / dT) + 1
    solution_slices = np.zeros((n_entries, len(u0) + len(v0)))
    solution_slices[0,:] = np.concatenate((u0, v0))

    u = np.copy(u0)
    v = np.copy(v0)
    for i in range(n_entries-1):
        u, v = fhn_euler_timestepper(u, v, dx, dt, dT, params, verbose=False)
        solution_slices[i,:] = np.concatenate((u, v))

    return solution_slices

def evolveTrajectory():
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    T = 2.0
    N = 200
    M = 2*N
    L = 20.0
    dt = 1.e-3
    dx = L / N
    n_initials = 500

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

if __name__ == '__main__':
    evolveTrajectory()