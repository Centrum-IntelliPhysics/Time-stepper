import numpy as np
import numpy.random as rd

from EulerTimestepper import euler_timestepper

def time_evolution(u0, T, dT, dt, dx, params):
    n_steps = int(T / dT)
    evolution = np.zeros((n_steps+1, len(u0)))
    u = np.copy(u0)
    evolution[0,:] = u

    for n in range(n_steps):
        u = euler_timestepper(u, dx, dt, dT, params)
        evolution[n+1,:] = u
    return evolution

def evolveDiffusionTrajectories():
    # Model parameters
    D = 0.1
    params = {'D': D}

    # Time-stepping parameters
    N = 50
    dx = 1.0 / N
    T = 100.0
    dT = 0.1
    dt = 0.001

    # Load the steady state and the leading eigenvectors
    u_ss = np.load('./Steady-State/euler_steady_state.npy')
    eigvecs = np.real(np.load('./Steady-State/Euler_eigenvectors.npy'))

    n_eigvecs = 10
    n_initials = 1000
    max_delta = 0.1
    rng = rd.RandomState()
    for n in range(n_initials):
        print('Initial #:', n+1)
        rand_index = rng.randint(0, n_eigvecs)
        u0 = u_ss + max_delta * rng.uniform(-1.0, 1.0) * eigvecs[:,rand_index]

        evolution = time_evolution(u0, T, dT, dt, dx, params)
        np.save('./data/trajectory_' + str(n) + '.npy', evolution[2:,:])

if __name__ == '__main__':
    evolveDiffusionTrajectories()