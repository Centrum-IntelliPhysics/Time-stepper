import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt

import matplotlib.pyplot as plt

from DeepONetWrapper import DeepONetWrapper
from EulerTimestepper import psi as euler_psi

def findSteadyStateNewtonGMRES():
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.09998733340349023
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    L = 20.0
    N = 200
    dx = L / N
    x_array = np.linspace(0.0, L, N)
    grid = x_array / L

    deeponet = DeepONetWrapper()
    flowmap = lambda x: np.concatenate(deeponet(x[0:N], x[N:], eps, grid))
    psi = lambda x: x - flowmap(x)

    # Find a good steady - state: start from a trainig point
    file = '/Users/hannesvdc/Research/Projects/Time-stepper/FitzHugh-Nagumo/data/multiparameter/FHN_BF_Evolution_eps=0p09998733340349023.npy'
    data = np.load(file)
    u0 = data[0,0,0:200]
    v0 = data[0,0,200:]
    x0 = np.concatenate((u0, v0))

    # Do Newton-GMRES on psi
    x_ss = opt.newton_krylov(psi, x0)
    u_ss = x_ss[0:N]
    v_ss = x_ss[N:]

    # Also plot the steady-state found by the Euler method
    T_psi = 1.0
    dt = 1.e-3
    x_euler = opt.newton_krylov(lambda x: euler_psi(x, T_psi, dx, dt, params), np.concatenate((u0, v0)))
    u_euler = x_euler[0:N]
    v_euler = x_euler[N:]

    # Plot the steady state
    plt.plot(x_array, u_ss, label='DeepONet Steady-State u(x)')
    plt.plot(x_array, v_ss, label='DeepONet Steady-State v(x)')
    plt.plot(x_array, u_euler, label='Euler Steady-State u(x)')
    plt.plot(x_array, v_euler, label='Euler Steady-State v(x)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    findSteadyStateNewtonGMRES()