import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import matplotlib.pyplot as plt

from DeepONetWrapper import DeepONetWrapper
from EulerTimestepper import fhn_euler_timestepper


def psi(x, T, dx, dt, params):
    N = x.size // 2
    u, v = x[0:N], x[N:]

    u_new, v_new = fhn_euler_timestepper(u, v, dx, dt, T, params)
    return np.concatenate((u - u_new, v - v_new)) / T # Not necessary to divide by T, but this works fine. Either way, T_psi = 1.0

def calculateSteadyState(u0, v0, dx, params):
    dt = 1.e-3
    T_psi = 1.0

    x0 = np.concatenate((u0, v0))
    F = lambda x: psi(x, T_psi, dx, dt, params)
    ss = opt.newton_krylov(F, x0)

    return ss[0:200], ss[200:]

def plotFitzHughNagumoSolution():
    # Model parameters
    L = 20.0
    N = 200
    dx = L / N
    x_array = np.linspace(0.0, L, N) / L

    # Initial condition: start from a training point and see
    # how it converges
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.09998733340349023
    file = '/Users/hannesvdc/Research/Projects/Time-stepper/FitzHugh-Nagumo/data/multiparameter/FHN_BF_Evolution_eps=0p09998733340349023.npy'
    data = np.load(file)
    u = data[0,0,0:200]
    v = data[0,0,200:]

    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}
    u0 = np.copy(u)
    v0 = np.copy(v)
    u_ss, v_ss = calculateSteadyState(u0, v0, dx, params)
    deeponet = DeepONetWrapper()

    #u = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    #v = sigmoid(x_array, 10, 0.0, 2.0, 0.1)

    # Timestepping
    dt = 1.0
    T = 10.0
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_array, u, label='u(x, t=0.0)')
    ax2.plot(x_array, v, label='v(x, t=0.0)')
    for n in range(int(T / dt)):
        u, v = deeponet.DeepONet(u, v, eps, x_array)

        ax1.plot(x_array, u, label='u(x, t='+str(n*dt)+')')
        ax2.plot(x_array, v, label='v(x, t='+str(n*dt)+')')
    ax1.plot(x_array, u_ss, label='Steady-State u(x)', linestyle='dashed')
    ax2.plot(x_array, v_ss, label='Steady-State u(x)', linestyle='dashed')

    # Plotting the final result
    ax1.set_title(r'$u(x,t)$')
    ax2.set_title(r'$v(x,t)$')
    ax1.legend()
    plt.show()

if __name__ == '__main__':
    plotFitzHughNagumoSolution()