import torch as pt
import numpy as np

from FNOWrapper import FNOWrapper


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
    file = '/Users/hannesvdc/Research/Projects/Time-stepper/FitzHugh-Nagumo/data/multiparameter/FHN_BF_Evolution_eps=0p09998733340349023.npy'
    data = pt.from_numpy(np.load(file))
    u0 = data[0,0,0:200]
    v0 = data[0,0,200:]

    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    wrapper = FNOWrapper()
    x0 = pt.concatenate((u0, v0))[None,:]
    print(x0.shape)
    print(wrapper(x0))

if __name__ == '__main__':
    runFNO()