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
    N = 200
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

    return solution_slices[2:,:] # Ignore t=0 and t=1

def evolveTrajectories():
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps_list = np.array([-0.01, 0.0, 0.005, 0.01, 0.013, 0.015, 0.017, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92])
    assert len(eps_list) == 20

    T = 450.0
    dt = 1.e-3
    dx = L / N
    dT = 1.0

    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'

    u_initials, v_initials = generateRandomInitials(plot=False)
    n_initials = u_initials.shape[0]
    for eps_index in range(len(eps_list)):
        eps = eps_list[eps_index]
        print('eps =', eps)
        params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

        eps_evolution = np.zeros((n_initials, int(T/dT)-1, 2*N)) # Ignore the first two timesteps
        for initial_index in range(n_initials):
            print('Initial ', initial_index+1)
            u0 = u_initials[initial_index,:]
            v0 = v_initials[initial_index,:]

            evolution = timeSimulation(u0, v0, dx, dt, T, dT, params)
            eps_evolution[initial_index,:,:] = evolution

        np.save(directory + 'FHN_Evolution_eps=' + str(round(eps,3)).replace('.', 'p') + '.npy', eps_evolution)     

if __name__ == '__main__':
    evolveTrajectories()