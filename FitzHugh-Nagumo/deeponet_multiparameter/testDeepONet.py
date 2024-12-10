import torch as pt
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from DeepONet import DeepONet
from EulerTimestepper import calculateSteadyState

pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float64)

# Load the model from file
p = 200
branch_layers = [400, 100, 100, 100, 100, 2*p]
trunk_layers = [2, 100, 100, 100, 100, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.load_state_dict(pt.load('./Results/model_deeponet_manyeps_fhn.pth', weights_only=True))

# Wrapper function that takes a general (u, v) input
L = 20.0
N = 200
deeponet_grid = pt.linspace(0.0, 1.0, N)
deeponet_grid_ext = deeponet_grid[:,None]
def deeponet(_x, _eps):
    input = pt.concatenate((pt.tile(_x, dims=(N, 1)), deeponet_grid_ext, _eps * pt.ones((N,1))), dim=1)
    output = network.forward(input)

    # Enforce boundary conditions
    output[0,:] = output[1,:]
    output[-1,:] = output[-2,:]

    # Return
    return pt.concatenate((output[:,0], output[:,1]))

def psi(x0, _eps, T_psi, dt):
    x = pt.from_numpy(np.copy(x0))
    n = int(T_psi / dt)
    for _ in range(n):
        x = deeponet(x, _eps)
    return x0 - x.numpy()

# Load the Initial Condition
eps = 0.1
data_directory = './../data/multiparameter/'
file = 'FHN_BF_Evolution_Initial=0_eps=' + str(eps).replace('.', 'p') + '_dT=0p001.npy'
data = np.load(data_directory + file)
rng = rd.RandomState()
u0 = pt.from_numpy(data[0,0:200] + 0.1 * rng.normal(0.0, 1.0, 200))
v0 = pt.from_numpy(data[0,200:]+ 0.1 * rng.normal(0.0, 1.0, 200))
x_array = L * deeponet_grid

# Find the steady-state of the Euler timestepper
dx = L / N
dt = 1.e-3
a0 = -0.03
a1 = 2.0
delta = 4.0
params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}
x0 = np.concatenate((u0.numpy(), v0.numpy()))
x_ss = calculateSteadyState(x0, 1.0, dx, dt, params)

# Do Timestepping 
T = 100.0
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x_array, u0, label=r'$T=0.0$')
ax2.plot(x_array, v0)
x = pt.concatenate((u0, v0))
for n in range(int(T / dt)+1):
    if n % 1000 == 0:
        print('t =', n * dt)
    x = deeponet(x, eps)
u = x[0:200]
v = x[200:]

# Find the DeepONet steady state
print('Initial Psi =', lg.norm(psi(x0, eps, 1.0, dt)))
x_nn_ss = opt.newton_krylov(lambda x: psi(x, eps, 1.0, dt), x0, f_tol=1.e-14, verbose=True)
np.save('./Results/DeepONet_steadystate_eps=' + str(eps).replace('.', 'p') + '.npy', x_nn_ss)

# Plot the timestepping and Newton-GMRES steady states
ax1.plot(x_array, u, label=r'$T='+str(n*dt)+'$ (DeepONet)')
ax2.plot(x_array, v)
ax1.plot(x_array, x_ss[0:200], label='Euler Steady State')
ax2.plot(x_array, x_ss[200:])
ax1.plot(x_array, x_nn_ss[0:200], label='DeepONet Steady State')
ax2.plot(x_array, x_nn_ss[200:])
ax1.set_title(r'$u(x)$')
ax2.set_title(r'$v(x)$')
ax1.set_xlabel(r'$x$')
ax2.set_xlabel(r'$x$')
ax1.legend()
plt.show()