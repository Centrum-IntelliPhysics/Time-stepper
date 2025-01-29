import sys
sys.path.append('../')

import numpy as np
import numpy.random as rd

from EulerTimestepper import fhn_euler_timestepper

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

def sampleEigenPerturbation():
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

    # Load the steady state condition and its eigenvectors
    directory = './../deeponet_singleparameter/Steady-State/'
    ss_file = 'euler_steady_state.npy'
    eigenvector_file = 'POD_eigenvectors.npy'
    x_ss = np.load(directory + ss_file)
    u_ss = x_ss[1,:]
    v_ss = x_ss[2,:]
    eigenvectors = np.load(directory + eigenvector_file)

    # Perturb the steady state in the direction of the eigenvectors and simulate
    rng = rd.RandomState(seed=100)
    n_eigenvectors = 10
    max_delta = 0.1
    n_initials = 1000
    for j in range(n_initials):
        print('Initial #', j+1)

        # Sample the Initial Condition
        eig_index = rng.randint(0, n_eigenvectors)
        delta_u = max_delta * rng.uniform(-1.0, 1.0)
        delta_v = max_delta * rng.uniform(-1.0, 1.0)
        u = u_ss + delta_u * np.real(eigenvectors[:N, eig_index])
        v = v_ss + delta_v * np.real(eigenvectors[N:, eig_index])

        # Time evolution: Ignore the first few time steps
        evolution = timeSimulation(u, v, dx, dt, T, dT, params)
        evolution = evolution[10:,:]

        # Store the time evolution
        np.save('./../data/singleparameter/FHN_SingleEpsilon_POD_Initial=' + str(j) + '_eps=' + str(eps).replace('.', 'p') + '_dT=' + str(dt).replace('.', 'p') + '.npy', evolution)

if __name__ == '__main__':
    sampleEigenPerturbation()
