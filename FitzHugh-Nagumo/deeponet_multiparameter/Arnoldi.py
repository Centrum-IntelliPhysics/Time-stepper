import torch as pt
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt

from DeepONet import DeepONet

pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float64)

N = 200
L = 20.0
dt = 0.001
T = 1.0
grid_ext = pt.linspace(0.0, 1.0, N)[:,None]

p = 200
branch_layers = [400, 100, 100, 100, 100, 2*p]
trunk_layers = [2, 100, 100, 100, 100, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.load_state_dict(pt.load('./Results/model_deeponet_manyeps_fhn.pth', weights_only=True))

def deeponet(x, eps):
    input = pt.concatenate((pt.tile(x, dims=(N, 1)), grid_ext, eps * pt.ones((N,1))), dim=1)
    output = network.forward(input)
    return pt.concatenate((output[:,0], output[:,1]))

def psi(x0, eps, T_psi, dt):
    x = pt.from_numpy(x0)
    n = int(T_psi / dt)
    for _ in range(n):
        x = deeponet(x, eps)
    return x0 - x.numpy()

def calculateEigenvalues():
    # Load the initial condition from file.
    eps0 = 0.1
    x0 = np.load('./Results/DeepONet_steadystate_eps=' + str(eps0).replace('.', 'p') + '.npy')

    # Calculate the eigenvalues of Psi in steady state
    print('\nCalculating Leading Eigenvalues of Psi using Arnoldi ...')
    T_psi = 1.0
    r_diff = 1.e-8
    d_psi_mvp = lambda w: (psi(x0 + r_diff * w, eps0, T_psi, dt) - psi(x0, eps0, T_psi, dt)) / r_diff
    D_psi = slg.LinearOperator(shape=(2*N, 2*N), matvec=d_psi_mvp)
    psi_eigvals = slg.eigs(D_psi, k=2*N-2, which='LM', return_eigenvectors=False)
    print('Done.')

    # Load eigenvalues of the right-hand side
    euler_eigvals = np.load('./Results/euler_eigenvalues_Tpsi=1p0.npy')
    f_eigvals = euler_eigvals[1,:]
    approx_deeponet_eigvals = 1.0 - np.exp(T_psi * f_eigvals)

    # Saving
    #toNumericString = lambda number: str(number).replace('.', 'p')
    #np.save(directory + 'euler_eigenvalues_Tpsi='+toNumericString(T_psi)+'.npy', np.vstack((psi_eigvals, f_eigvals, psi_approx_eigvals)))

    # Plot the Eigenvalues
    plt.scatter(np.real(psi_eigvals), np.imag(psi_eigvals), label=r'Eigenvalues $\mu$ of $\psi$ ')
    plt.scatter(np.real(approx_deeponet_eigvals), np.imag(approx_deeponet_eigvals), label=r'$1 - \exp\left(\sigma T\right)$ ')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(visible=True, which='major', axis='both')
    plt.title('Timestepper Eigenvalues')
    plt.legend()

    plt.figure()
    plt.scatter(np.real(f_eigvals), np.imag(f_eigvals), label=r'Eigenvalues $\sigma$ of $f$ ')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(visible=True, which='major', axis='both')
    plt.title('PDE Eigenvalues')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    calculateEigenvalues()
