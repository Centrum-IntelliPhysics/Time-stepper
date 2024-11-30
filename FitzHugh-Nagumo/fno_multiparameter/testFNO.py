import torch as pt
import numpy as np

import matplotlib.pyplot as plt

from FNOWrapper import FNOWrapper

pt.set_default_dtype(pt.float)
pt.set_grad_enabled(False)

def runFNO():
    # Model parameters
    L = 20.0
    N = 200
    dx = L / N
    x_array = pt.linspace(0.0, L, N) / L

    # Initial condition: start from a training point and see
    # how it converges
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.09998733340349023
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Load the Initial Condition
    removeDot = lambda r: str(r).replace('.', 'p')
    file = '/Users/hannesvdc/Research/Projects/Time-stepper/FitzHugh-Nagumo/data/multiparameter/FHN_BF_Evolution_eps=' + removeDot(eps) + '.npy'
    data = pt.from_numpy(np.load(file)).to(dtype=pt.float)
    u = data[0,0,0:200]
    v = data[0,0,200:]

    # Do Timestepping based on the FNO network
    wrapper = FNOWrapper()
    psi = lambda x: x - np.concatenate(wrapper(x[0:N], x[N:], eps, x_array))

    dt = 1.0
    T = 10.0
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_array, u, label=r'$t=0.0$')
    ax2.plot(x_array, v, label=r'$t=0.0$')
    #psi_list = [pt.norm(psi(np.concatenate((u0, v0))))]
    for n in range(int(T / dt)):
        u_new, v_new = wrapper(u, v, eps)
        #psi_list.append(pt.norm(psi(np.concatenate((u_new, v_new)))))

        u = pt.clone(u_new)
        v = pt.clone(v_new)

        ax1.plot(x_array, u, label=r'$t='+str(n*dt)+'$')
        ax2.plot(x_array, v, label=r'$t='+str(n*dt)+'$')

    # Value of psi
    #print('psi deeponet', pt.norm(psi(np.concatenate((u, v)))))

    # Plotting the final result
    ax1.set_title(r'$u(x,t)$')
    ax2.set_title(r'$v(x,t)$')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

if __name__ == '__main__':
    runFNO()