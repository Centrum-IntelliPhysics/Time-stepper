import torch as pt
import numpy as np
import matplotlib.pyplot as plt

from DeepONet import DeepONet

pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float64)

# Load the model from file
p = 200
branch_layers = [3, 100, 100, 100, 100, 2*p]
trunk_layers = [1, 100, 100, 100, 100, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.load_state_dict(pt.load('./Results/model_deeponet_fhn.pth', weights_only=True))

# Wrapper function that takes a general (u, v) input
L = 20.0
N = 200
deeponet_grid = pt.linspace(0.0, 1.0, N)
deeponet_grid_ext = deeponet_grid[:,None]
def deeponet(_u, _v, _eps):
    input = pt.concatenate((_u[:,None], _v[:,None], _eps * pt.ones_like(deeponet_grid_ext), deeponet_grid_ext), dim=1)
    print('Input Shape', input.shape)
    output = network.forward(input)
    return output[:,0], output[:,1]
psi = lambda _u, _v, _eps: (_u, _v) - deeponet(_u, _v, _eps)

# Load the Initial Condition
eps = 0.09998733340349023
file = '/Users/hannesvdc/Research/Projects/Time-stepper/FitzHugh-Nagumo/data/multiparameter/FHN_BF_Evolution_eps=' + str(eps).replace('.', 'p') + '.npy'
data = np.load(file)
u = pt.from_numpy(data[0,0,0:200])
v = pt.from_numpy(data[0,0,200:])
x_array = L * deeponet_grid

# Do Timestepping 
dt = 1.0
T = 10.0
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x_array, u, label=r'$t=0.0$')
ax2.plot(x_array, v, label=r'$t=0.0$')
psi_list = [pt.norm(psi(u, v, eps))]
for n in range(int(T / dt)):
    u, v = deeponet(u, v, eps)
    psi_list.append(pt.norm(psi(u, v, eps)))

    ax1.plot(x_array, u, label=r'$t='+str(n*dt)+'$')
    ax2.plot(x_array, v, label=r'$t='+str(n*dt)+'$')

ax1.legend()
plt.show()