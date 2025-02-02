import sys
sys.path.append('../')

from matplotlib import pyplot as plt
import numpy as np
import numpy.random as rd

from EulerTimestepper import fhn_euler_timestepper, calculateLeadingEigenvalues

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

# Start from points on the bifurcation diagram, take some values of epsilon, 
# Perturb them slightly, and evolve over given time window.
def sampleEigenPerturbation():
    # Model parameters
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    params = {'delta': delta, 'a0': a0, 'a1': a1}

    # Geometry parameters
    L = 20.0
    N = 200
    M = 2 * N
    dx = L / N

    # Timestepping parameters
    T = 20.0
    dt = 1.e-3
    dT = 0.1
    n_initials = 200
    
    # Load the bifurcation diagram
    load_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
    store_directory = './../data/multiparameter/'
    bf_data = np.load(load_directory + 'euler_bf_diagram.npy')
    euler_p1 = bf_data[:,0:M]
    euler_eps1 = bf_data[:,M]
    euler_p2 = np.flip(bf_data[:,M+1:2*M+1])
    euler_eps2 = np.flip(bf_data[:,2*M+1], axis=0)
    euler_eps = np.concatenate((euler_eps2, euler_eps1))
    euler_path = np.vstack((euler_p2, euler_p1))

    # Determine the epsilon values to sample. Uses the fact that the arrays are monotonous
    def find_nearest(array, value, start_index):
        index = start_index
        c_value = array[start_index]
        while index < len(array):
            if np.abs(array[index] - value) > np.abs(c_value - value):
                return index
            else:
                c_value = array[index]
            index += 1
        return None
    eps_values = [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Run through the epsilon values, find the point on the bf diagram nearest to it, compute its POD, perturb, and evolve.
    rng = rd.RandomState(seed=100)
    running_index = 0
    for eps_index in range(len(eps_values)):
        eps = eps_values[eps_index]
        print('\nEps =', eps)

        # 1. Get the closest poin on the bifurcation diagram
        print('Getting point on the bifurcation diagram')
        running_index = find_nearest(euler_eps, eps, running_index)
        _eps = euler_eps[running_index]
        u0 = euler_path[running_index, :N]
        v0 = euler_path[running_index, N:]
        x0 = np.concatenate((u0, v0))
        print('Closes point on the BF diagram:', eps, _eps)

        # 2. Compute the complete Jacobian and do POD
        print('Computing POD')
        params['eps'] = _eps
        T_psi = 1.0
        _, eigvecs = calculateLeadingEigenvalues(x0, T_psi, dx, dt, params, k=10)

        # 3. Perturb the initial condition 1000 times
        for initial in range(n_initials):
            print('Evolving Initial #:', initial)
            eig_index = rng.randint(0, 10)
            delta_u = rng.uniform(-0.1, 0.1)
            delta_v = rng.uniform(-0.1, 0.1)
            u = u0 + delta_u * np.real(eigvecs[:N, eig_index])
            v = v0 + delta_v * np.real(eigvecs[N:, eig_index])

            # 4. Evolve the perturbed initial condition
            evolution = timeSimulation(u, v, dx, dt, T, dT, params)

            # 5. Store to file
            np.save(store_directory + 'FHN_MultiEps_Evolution_Initial=' + str(initial) + '_eps=' + str(eps).replace('.', 'p') + '.npy', evolution)

if __name__ == '__main__':
    sampleEigenPerturbation()