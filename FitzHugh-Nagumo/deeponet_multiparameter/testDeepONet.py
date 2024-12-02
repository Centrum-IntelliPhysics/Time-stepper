import torch as pt
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from DeepONet import DeepONet
from EulerTimestepper import fhn_euler_timestepper

pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float64)

def psi_euler(x, T, dx, dt, params):
    N = x.size // 2
    u, v = x[0:N], x[N:]

    u_new, v_new = fhn_euler_timestepper(u, v, dx, dt, T, params)
    return np.concatenate((u - u_new, v - v_new)) / T # Not necessary to divide by T, but this works fine. Either way, T_psi = 1.0

def calculateSteadyState(u0, v0, dx, params):
    dt = 1.e-3
    T_psi = 1.0

    x0 = np.concatenate((u0, v0))
    F = lambda x: psi_euler(x, T_psi, dx, dt, params)
    ss = opt.newton_krylov(F, x0)

    return ss[0:200], ss[200:]

# Load the model from file
p = 200
branch_layers = [401, 100, 100, 100, 100, 2*p]
trunk_layers = [1, 100, 100, 100, 100, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.load_state_dict(pt.load('./Results/model_deeponet_fhn.pth', weights_only=True))

# Wrapper function that takes a general (u, v) input
L = 20.0
N = 200
deeponet_grid = pt.linspace(0.0, 1.0, N)
deeponet_grid_ext = deeponet_grid[:,None]
def deeponet(_u, _v, _eps):
    input_row = pt.concatenate((_u, _v, pt.tensor([_eps])))
    input = pt.concatenate((pt.tile(input_row, dims=(N,1)), deeponet_grid_ext), dim=1)
    output = network.forward(input)
    return output[:,0], output[:,1]
def psi(_u, _v, _eps):
    u_new, v_new = deeponet(_u, _v, _eps)
    return _u - u_new, _v - v_new

# Load the Initial Condition
eps = 0.09998733340349023
file = '/Users/hannesvdc/Research/Projects/Time-stepper/FitzHugh-Nagumo/data/multiparameter/FHN_BF_Evolution_eps=' + str(eps).replace('.', 'p') + '.npy'
data = np.load(file)
u = pt.from_numpy(data[0,0,0:200])
v = pt.from_numpy(data[0,0,200:])
x_array = L * deeponet_grid

# Find the steady-state of the Euler timestepper
dx = L / N
a0 = -0.03
a1 = 2.0
delta = 4.0
params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}
u0 = np.copy(u.numpy())
v0 = np.copy(v.numpy())
u_ss, v_ss = calculateSteadyState(u0, v0, dx, params)

# Find the steady-state of the DeepONet
F_deeponet = lambda x: pt.concatenate(psi(pt.from_numpy(x[0:N]), pt.from_numpy(x[N:]), eps)).numpy()
x0 = np.concatenate((u0, v0))
deeponet_ss = opt.newton_krylov(F_deeponet, x0)

# Do Timestepping 
dt = 1.0
T = 450.0
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x_array, u, label=r'$t=0.0$')
ax2.plot(x_array, v)
psi_list = [pt.norm(pt.concatenate(psi(u, v, eps)))]
for n in range(int(T / dt)):
    u, v = deeponet(u, v, eps)
    psi_list.append(pt.norm(pt.concatenate(psi(u, v, eps))))
    print('psi', psi_list[-1])

ax1.plot(x_array, u, label=r'$t='+str(n*dt)+'$')
ax2.plot(x_array, v)
ax1.plot(x_array, u_ss, label='Euler Newton-GMRES Steady State')
ax2.plot(x_array, v_ss)
ax1.plot(x_array, deeponet_ss[0:N], label='DeepONet Newton-GMRES Steady State')
ax2.plot(x_array, deeponet_ss[N:])
ax1.set_title(r'$u(x)$')
ax2.set_title(r'$v(x)$')
ax1.set_xlabel(r'$x$')
ax2.set_xlabel(r'$x$')
ax1.legend()
plt.show()