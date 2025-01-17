import torch as pt
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
import math
from DeepONet import DeepONet

pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float32)

N = 200
L = 20.0
dt = 0.001
dT = 100 * dt
grid_ext = pt.linspace(0.0, 1.0, N)[:,None]

p = 400
branch_input_size = 400
trunk_input_size = 1
branch_layers = [branch_input_size, 400, 400, 400, 400, 400, 2*p]
trunk_layers  = [trunk_input_size,  400, 400, 400, 400, 400, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.load_state_dict(pt.load('./Results/model_deeponet_fhn_ss.pth', weights_only=True))

# Wrapper function that takes a general (u, v) input
L = 20.0
N = 200
deeponet_grid = pt.linspace(0.0, 1.0, N)
deeponet_grid_ext = deeponet_grid[:,None]
def deeponet(x):
    input = pt.concatenate((pt.tile(x, dims=(N,1)), deeponet_grid_ext), dim=1)
    output = network.forward(input)
    return pt.concatenate((output[:,0], output[:,1]))

# Calculate the deeponet steady state using Newton-GMRES
def psi(x, T_psi):
    x0 = pt.clone(x)
    n = int(T_psi / dT)
    for _ in range(n):
        x = deeponet(x)
    return x0 - x

def debugFDandAD():
    # Load the initial condition from file.
    w_ss = pt.tensor(np.load('./Results/DeepONet_steadystate.npy'), dtype=pt.float32)

    # Calculate psi-value in the steady state
    T_psi = 1.0
    input = pt.clone(w_ss).requires_grad_(True)
    output = psi(input, T_psi)
    print('psi norm', pt.norm(output))

    eps_fd = 1.e-8
    d_psi = lambda v: (psi(w_ss + eps_fd * v, T_psi) - output) / eps_fd
    for n in range(2*N):
        e_n = pt.eye(2*N)[:,n]
        print(e_n.shape)

        # AD
        grad_ad = pt.autograd.grad(outputs=output, inputs=input, grad_outputs=(e_n), retain_graph=True)[0]
        #print(grad_ad)

        # FD
        grad_fd = d_psi(e_n)

        print(n, ': ', pt.norm(grad_ad - grad_fd)/pt.norm(grad_ad))

def calculateEigenvalues():
    f_info = pt.finfo(pt.float32)
    eps_fd = math.sqrt(f_info.eps)

    # Load the initial condition from file.
    w_ss = pt.tensor(np.load('./Results/DeepONet_steadystate.npy'), dtype=pt.float32)

    # Calculate psi-value in the steady state
    T_psi = 1.0
    input = pt.clone(w_ss).requires_grad_(True)
    output = psi(input, T_psi)

    # Setup the Analytic Jacobian Matrix
    print('\nComputing the Analytic Jacobian Matrix ...')
    dF_ad = np.zeros((2*N, 2*N))
    for n in range(2*N):
        grad_output = pt.zeros_like(output)
        grad_output[n] = 1.0

        grad_n = pt.autograd.grad(outputs=output, inputs=input, grad_outputs=grad_output, retain_graph=True)[0]
        dF_ad[:,n] = grad_n.detach().numpy()

    # Setup the finite-difference Jacobian Matrix
    print('\nComputing the Numerical Jacobian Matrix ...')
    print('Sqrt Machine Precision:', eps_fd)
    dF_fd = np.zeros((2*N, 2*N))
    d_psi = lambda v: pt.norm(v) * (psi(w_ss + eps_fd * v / pt.norm(v), T_psi) - psi(w_ss, T_psi)) / eps_fd
    for n in range(2*N):
        e_n = pt.zeros(2*N)
        e_n[n] = 1.0
        dF_fd[:,n] = d_psi(e_n).detach().numpy()

    print('Running QR Method')
    eigvals_ad, eigvecs_ad = lg.eig(dF_ad)
    eigvals_fd, eigvecs_fd = lg.eig(dF_fd)

    # Load eigenvalues of the right-hand side
    euler_eigvals = np.load('./Steady-State/euler_eigenvalues.npy')
    f_eigvals = euler_eigvals[1,:]
    approx_deeponet_eigvals = 1.0 - np.exp(T_psi * f_eigvals)

    # Plot the Eigenvalues
    plt.scatter(np.real(eigvals_fd), np.imag(eigvals_fd), label='Finite Differences')
    plt.scatter(np.real(eigvals_ad), np.imag(eigvals_ad), edgecolors='tab:orange', facecolor='none', label='Automatic Differentiation')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(visible=True, which='major', axis='both')
    plt.title('DeepONet Eigenvalues')
    plt.legend()

    # Plot the Eigenvalues
    plt.figure()
    plt.scatter(np.real(approx_deeponet_eigvals), np.imag(approx_deeponet_eigvals), label='PDE')
    plt.scatter(np.real(eigvals_ad), np.imag(eigvals_ad), label='DeepONet')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(visible=True, which='major', axis='both')
    plt.title('DeepONet vs PDE Eigenvalues')
    plt.legend()

    plt.figure()
    plt.scatter(np.real(f_eigvals), np.imag(f_eigvals), label=r'Eigenvalues $\sigma$ of $\nabla f$ ')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(visible=True, which='major', axis='both')
    plt.title('PDE Eigenvalues')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    calculateEigenvalues()
