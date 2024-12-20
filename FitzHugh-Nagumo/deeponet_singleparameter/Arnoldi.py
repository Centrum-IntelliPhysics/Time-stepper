import torch as pt
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt

from DeepONet import DeepONet

pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float32)

N = 200
L = 20.0
dt = 0.001
dT = 10 * dt
grid_ext = pt.linspace(0.0, 1.0, N)[:,None]

p = 200
branch_input_size = 400
trunk_input_size = 1
branch_layers = [branch_input_size, 400, 400, 400, 400, 2*p]
trunk_layers  = [trunk_input_size,  400, 400, 400, 400, 2*p]
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
def psi(x0, T_psi):
    x = pt.Tensor(x0)

    n = int(T_psi / dT)
    for _ in range(n):
        x = deeponet(x)
    return x0 - x.numpy()

def calculateEigenvalues():
    # Load the initial condition from file.
    x0 = np.load('./Results/DeepONet_steadystate.npy')

    # Calculate the eigenvalues of Psi in steady state
    print('\nCalculating Leading Eigenvalues of Psi using Arnoldi ...')
    T_psi = 1.0
    r_diff = 1.e-8
    d_psi_mvp = lambda w: (psi(x0 + r_diff * w, T_psi) - psi(x0, T_psi)) / r_diff
    D_psi = slg.LinearOperator(shape=(2*N, 2*N), matvec=d_psi_mvp)
    #psi_eigvals = slg.eigs(D_psi, k=2*N-2, which='SM', return_eigenvectors=False)
    print('Done.')

    # Calculate the eigenvalues using the Francis QR method
    A = np.zeros((2*N, 2*N))
    I = np.eye(2*N)
    for row in range(2*N):
        print(row)
        A[:,row] = D_psi.matvec(I[:,row])
    print('Running QR Method')
    psi_eigvals = lg.eigvals(A)

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
    plt.title(r'DeepONet $\psi$ Eigenvalues')
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
