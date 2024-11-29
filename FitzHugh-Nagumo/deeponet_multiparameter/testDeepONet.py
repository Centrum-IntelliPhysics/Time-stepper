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
    psi = lambda x: x - np.concatenate(deeponet(x[0:N], x[N:], eps, x_array))

    # Timestepping
    dt = 1.0
    T = 450.0
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_array, u, label=r'$t=0.0$')
    ax2.plot(x_array, v, label=r'$t=0.0$')
    psi_list = [lg.norm(psi(np.concatenate((u0, v0))))]
    for n in range(int(T / dt)):
        u_new, v_new = deeponet(u, v, eps, x_array)
        psi_list.append(lg.norm(psi(np.concatenate((u_new, v_new)))))

        u = np.copy(u_new)
        v = np.copy(v_new)

        ax1.plot(x_array, u, label=r'$t='+str(n*dt)+'$')
        ax2.plot(x_array, v, label=r'$t='+str(n*dt)+'$')

    # Value of psi
    psi = lambda x: x - np.concatenate(deeponet(x[0:N], x[N:], eps, x_array))
    print('psi deeponet', lg.norm(psi(np.concatenate((u, v)))))

    ax1.plot(x_array, u_ss, label='Steady-State', linestyle='dashed')
    ax2.plot(x_array, v_ss, label='Steady-State', linestyle='dashed')

    # Plotting the final result
    ax1.set_title(r'$u(x,t)$')
    ax2.set_title(r'$v(x,t)$')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.figure()
    t_list = np.arange(0.0, len(psi_list))
    plt.semilogy(t_list, psi_list, label=r'$\psi(u(t), v(t)$')
    plt.xlabel(r'$t [s]$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plotFitzHughNagumoSolution()