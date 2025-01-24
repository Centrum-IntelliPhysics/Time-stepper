import torch as pt
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from DeepONet import DeepONet
from EulerTimestepper import calculateSteadyState, psi

# Just some sanity pytorch settings
pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float32)

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up DeepONet Neural Net...')
p = 400
branch_input_size = 50
trunk_input_size = 1
branch_layers = [branch_input_size, 400, 400, 400, 400, p]
trunk_layers  = [trunk_input_size,  400, 400, 400, 400, p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.load_state_dict(pt.load('./Results/model_deeponet_fhn.pth', weights_only=True))
pt.save(network.state_dict(), './Results/model_deeponet_fhn_ss.pth')

# Wrapper function that takes a general (u, v) input
N = 50
deeponet_grid = pt.linspace(0.0, 1.0, N)
deeponet_grid_ext = deeponet_grid[:,None]
def deeponet(x): # x and output are numpy arrays. Only use Tensor within the neural net
    input = pt.concatenate((pt.tile(pt.tensor(x), dims=(N,1)), deeponet_grid_ext), dim=1)
    output = network.forward(input)
    return np.copy(output[:,0].numpy())

# Load the Initial Condition
rng = rd.RandomState()
initial_index = 474
data = np.load('./data/trajectory_' + str(initial_index) + '.npy').astype(np.float32)
u0 = data[0,:]

# Load of the Euler timestepper
print('\nComputing the Euler Steady State ...')
dx = 1.0 / N
dt = 1.e-3
dT = 100 * dt
T_psi = 1.0
D = 0.1
params = {'D': D}
x_ss = calculateSteadyState(u0, T_psi, dx, dt, params)
print('Euler Steady-State Found, Final Psi:', lg.norm(psi(x_ss, T_psi, dx, dt, params)))

# Do Timestepping 
T = 100.0
x_array = np.linspace(0.0, 1.0, N)
plt.plot(x_array, u0, label=r'$T=0.0$')
u = np.copy(u0)
for n in range(1, int(T / dT)+1):
    if n % 1000 == 0:
        print('t =', n * dT)
    u = deeponet(u)

# Calculate the deeponet steady state using Newton-GMRES
def deeponet_psi(x0, T_psi):
    x = np.copy(x0)

    n = int(T_psi / dT)
    for _ in range(n):
        x = deeponet(x)
    return x0 - x
x_nn_ss = opt.newton_krylov(lambda x: deeponet_psi(x, T_psi), u0, f_tol=1.e-6, verbose=True, method='gmres')
np.save('./Results/DeepONet_steadystate.npy', x_nn_ss)

plt.plot(x_array, u, label=r'$T=100.0$')
plt.plot(x_array, x_ss, label='Euler Steady State')
plt.plot(x_array, x_nn_ss, label='DeepONet Steady State')
plt.title(r'$u(x)$')
plt.xlabel(r'$x$')
plt.legend(loc='upper left')
plt.show()