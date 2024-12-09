import sys
sys.path.append('../deeponet_multiparameter/')

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from EulerTimestepper import fhn_euler_timestepper

L = 20.0
N = 200

# Original sigmoid between 0 and 1. To make it between -1 and 1, shift by y_center=-0.5 and y_scale=2
def sigmoid(x, x_center=0.0, y_center=0.0, x_scale=1.0, y_scale=1.0):
    return y_scale / (1.0 + np.exp(-(x  - x_center)/x_scale)) + y_center

def generateRandomInitials(plot=False):
    x_array = np.linspace(0.0, L, N)

    n_initials = 20
    rng = rd.RandomState(seed=100)
    u_shifts = rng.uniform(4.0, 16.0, size=n_initials)
    u_scale = rng.uniform(0.5, 2, size=n_initials)
    v_shifts = rng.uniform(5.0, 15.0, size=n_initials)
    v_scale = rng.uniform(0.5, 2, size=n_initials)

    u_initials = np.zeros((n_initials, N))
    v_initials = np.zeros((n_initials, N))
    for i in range(n_initials):
        u0 = sigmoid(x_array, u_shifts[i], -1.0, u_scale[i], 2.0)
        v0 = sigmoid(x_array, v_shifts[i], -1.0, v_scale[i], 2.0)
        u_initials[i,:] = u0
        v_initials[i,:] = v0

    # Plot all initial conditions in two separate figures
    if plot:
        plt.plot(x_array, u_initials.T, color='blue')
        plt.legend([r'Random Initials $u(x, t=0)$'])
        plt.figure()
        plt.plot(x_array, v_initials.T, color='orange')
        plt.legend([r'Random Initials $v(x, t=0)$'])
        plt.show()

    return u_initials, v_initials

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
    dt = 1.e-3
    dx = L / N
    n_initials = 20

    # Load the bifurcation diagram to determine a good initial point
    eps = 0.1
    load_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
    bf_data = np.load(load_directory + 'euler_bf_diagram.npy')
    x0 = bf_data[0,0:M]
    u0 = x0[0:N]
    v0 = x0[N:]

    store_directory = './../data/multiparameter/'
    rng = rd.RandomState()
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}
    for j in range(n_initials):
        print('initial ', j)
        u = u0 + 0.1*rng.normal(0.0, 1.0, N)
        v = v0 + 0.1*rng.normal(0.0, 1.0, N)
        evolution = timeSimulation(u, v, dx, dt, T, dt, params)

        # Store the time evolution
        np.save(store_directory + 'FHN_BF_Evolution_Initial=' + str(j) + '_eps=' + str(eps).replace('.', 'p') + '_dT=' + str(dt).replace('.', 'p') + '.npy', evolution)

# Start from points on the bifurcation diagram, take some values of epsilon, 
# Perturb them slightly, and evolve over given time window.
def evolveBFTrajectories():
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    T = 50.0
    N = 200
    M = 2*N
    dt = 1.e-3
    dx = L / N
    dT = 1.0
    n_initials = 20

    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
    bf_data = np.load(directory + 'euler_bf_diagram.npy')

    N_points = bf_data.shape[0]
    euler_p1 = bf_data[:,0:M]
    euler_eps1 = bf_data[:,M]
    euler_p2 = bf_data[:,M+1:2*M+1]
    euler_eps2 = bf_data[:,2*M+1]
    print(euler_eps1.shape)

    # Take 10 values of epsilon from each branch.
    max_index = int(2.0/3.0* N_points)
    indices = np.linspace(1, max_index, 10, dtype=int)
    print(indices)
    rng = rd.RandomState(seed=100)
    for i in range(len(indices)):
        eps = euler_eps1[indices[i]]
        print('eps =', eps)
        params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}
        u0 = euler_p1[indices[i], 0:200]
        v0 = euler_p1[indices[i], 200:]

        eps_evolution = np.zeros((n_initials, int(T/dT)-1, 2*N)) # Ignore the first two timesteps
        for j in range(n_initials):
            print('initial ', j)
            u = u0 + 0.01*rng.normal(0.0, 1.0, N)
            v = v0 + 0.01*rng.normal(0.0, 1.0, N)
            evolution = timeSimulation(u, v, dx, dt, T, dT, params)
            eps_evolution[j,:,:] = evolution

        np.save(directory + 'DeepONet Data/FHN_BF_Evolution_eps=' + str(eps).replace('.', 'p') + '.npy', eps_evolution)

    for i in range(len(indices)):
        eps = euler_eps2[indices[i]]
        print('eps =', eps)
        params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}
        u0 = euler_p2[indices[i], 0:200]
        v0 = euler_p2[indices[i], 200:]

        eps_evolution = np.zeros((n_initials, int(T/dT)-1, 2*N)) # Ignore the first two timesteps
        for j in range(n_initials):
            print('initial ', j)
            u = u0 + 0.1*rng.normal(0.0, 1.0, N)
            v = v0 + 0.1*rng.normal(0.0, 1.0, N)
            evolution = timeSimulation(u, v, dx, dt, T, dT, params)
            eps_evolution[j,:,:] = evolution

        np.save(directory + 'DeepONet Data/FHN_BF_Evolution_eps=' + str(eps).replace('.', 'p') + '.npy', eps_evolution)

if __name__ == '__main__':
    evolveTrajectory()