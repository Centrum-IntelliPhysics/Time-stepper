import sys
sys.path.append('../')

import torch as pt
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from DeepONet import DeepONet
from EulerTimestepper import calculateSteadyState, psi

pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float32)

# Load the model from file
p = 400
branch_input_size = 400
trunk_input_size = 1
branch_layers = [branch_input_size, 400, 400, 400, 400, 400, 2*p]
trunk_layers  = [trunk_input_size,  400, 400, 400, 400, 400, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.load_state_dict(pt.load('./Results/model_deeponet_fhn.pth', weights_only=True))
pt.save(network.state_dict(), './Results/model_deeponet_fhn_ss.pth')

# Wrapper function that takes a general (u, v) input
L = 20.0
N = 200
deeponet_grid = pt.linspace(0.0, 1.0, N)
deeponet_grid_ext = deeponet_grid[:,None]
def deeponet(x):
    input = pt.concatenate((pt.tile(x, dims=(N,1)), deeponet_grid_ext), dim=1)
    output = network.forward(input)
    return pt.concatenate((output[:,0], output[:,1]))

# Load the Initial Condition
eps = 0.1
rng = rd.RandomState()
initial_index = 474
data_directory = '../data/singleparameter/'
file = 'FHN_SingleEpsilon_POD_Initial=' + str(initial_index) + '_eps=0p1_dT=0p001.npy'
data = np.load(data_directory + file)
u = pt.Tensor(data[0,0:200])
v = pt.Tensor(data[0,200:])
x_array = L * deeponet_grid

# Load of the Euler timestepper
print('\nComputing the Euler Steady State ...')
dx = L / N
dt = 1.e-3
dT = 100 * dt
a0 = -0.03
a1 = 2.0
delta = 4.0
params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}
x0 = np.concatenate((u.numpy(), v.numpy()))
x_ss = calculateSteadyState(x0, 1.0, dx, dt, params)
print('Euler Steady-State Found, Final Psi:', lg.norm(psi(x_ss, 1.0, dx, dt, params)))

# Do Timestepping and store the intermediate solutions for pcolor
# Start from a 'bad' initial sigmoid shape
def sigmoid(x_array, x_center=0.0, y_center=0.0, x_scale=1.0, y_scale=1.0):
    return y_scale / (1.0 + np.exp(-(x_array  - x_center)/x_scale)) + y_center

u0 = 0.1*pt.tensor(sigmoid(x_array.numpy(), 6.0, -1, 1.0, 2.0))
v0 = 0.1*pt.tensor(sigmoid(x_array.numpy(), 10, 0.0, 2.0, 0.1))
T = 10.0
ax1 = plt.gca()
x = pt.concatenate((u0, v0))
full_field = np.zeros((int(T/dT+1), 400))
full_field[0,:] = np.copy(x.numpy())
for n in range(1, int(T / dT)+1):
    if n % 1000 == 0:
        print('t =', n * dT)
    x = deeponet(x)
    full_field[n,:] = np.copy(x.numpy())
u = x[0:200]
v = x[200:]

# Calculate the deeponet steady state using Newton-GMRES
def deeponet_psi(x0, T_psi):
    x = pt.tensor(x0)

    n = int(T_psi / dT)
    for _ in range(n):
        x = deeponet(x)
    return x0 - x.numpy()
x_nn_ss = opt.newton_krylov(lambda x: deeponet_psi(x, 1.0), x0, f_tol=1.e-6, verbose=True, method='gmres')
np.save('./Results/DeepONet_steadystate.npy', x_nn_ss)

ax1.plot(x_array, x_ss[:200], color='blue', label=r'Euler FD $u$')
ax1.plot(x_array, x_ss[200:], color='red', label=r'FD Euler $v$')
ax1.plot(x_array, x_nn_ss[:200], color = "#e9a92f", linestyle='--', label=r'DeepONet $u$')
ax1.plot(x_array, x_nn_ss[200:], color='purple', linestyle='--', label=r'DeepONet $v$')
#ax1.plot(x_array, u, color = "#e9a92f", linestyle='--', label=r'DeepONet $T$')
#ax1.plot(x_array, v, color='purple', linestyle='--', label=r'DeepONet $T$')
ax1.set_xlabel(r'$x$')
ax1.legend(loc='upper left')
x_grid_locs = [5, 10, 15]
y_grid_locs = [ -0.5, 0, 0.5]
ax1.set_xticks(x_grid_locs)
ax1.set_yticks(y_grid_locs)
ax1.grid(True, alpha=0.7)
ax1.set_xlim((0, 20))
ax1.set_ylim((-1,1))

plt.figure()
print(full_field[:,0:200])
X = np.linspace(0.0, L, 200)
T = np.linspace(0.0, T, int(T / dT)+1)
X, T = np.meshgrid(X, T)
vmin = np.min(full_field[:,0:200])
vmax = np.max(full_field[:,0:200])
plt.pcolor(X, T, full_field[:,0:200], vmin=vmin, vmax=vmax, cmap='jet')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.show()