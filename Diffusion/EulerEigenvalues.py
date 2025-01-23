import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

from EulerTimestepper import calculateSteadyState, psi, pde_rhs

directory = './Steady-State/'

D = 0.1
params = {'D': D}
N = 50
x_array = np.linspace(0.0, 1.0, N)
dx = 1.0 / N
u0 = 1.0 + np.sin(0.5 * np.pi * x_array) # Dirichlet boundary conditions u0(t,0)=1, u0(t,1)=2

# Compute the steady state
T_psi = 1.0
dt = 1.e-3
u_ss = calculateSteadyState(u0, T_psi, dx, dt, params)
np.save(directory + 'euler_steady_state.npy', u_ss)

# Build the Jacobian in steady-state
eps_fd = 1.e-8
psi_matvec = lambda v: (psi(u_ss + eps_fd * v, T_psi, dx, dt, params) - psi(u_ss, T_psi, dx, dt, params)) / eps_fd
J_psi = np.zeros((N, N))
for n in range(N):
    e_n = np.zeros(N)
    e_n[n] = 1.0
    J_psi[:,n] = psi_matvec(e_n)

# Calculate its eigenvalues and eigenvectors
psi_eigenvalues, psi_eigenvectors = lg.eig(J_psi)
sorted_indices = np.argsort(np.abs(psi_eigenvalues))
psi_eigenvalues = psi_eigenvalues[sorted_indices]
psi_eigenvectors = psi_eigenvectors[:,sorted_indices]
np.save(directory + 'Euler_eigenvalues.npy', psi_eigenvalues)
np.save(directory + 'Euler_eigenvectors.npy', psi_eigenvectors)

# Also compute the eigenvalues of the right-hand side f(u)
f_matvec = lambda v: (pde_rhs(u_ss + eps_fd * v, dx, params) - pde_rhs(u_ss, dx, params))/ eps_fd
J_f = np.zeros((N, N))
for n in range(N):
    e_n = np.zeros(N)
    e_n[n] = 1.0
    J_f[:,n] = f_matvec(e_n)

# Calculate its eigenvalues and eigenvectors
f_eigenvalues, f_eigenvectors = lg.eig(J_f)
approx_psi_eigenvalues = 1.0 - np.exp(T_psi * f_eigenvalues)

# Plot
plt.scatter(np.real(f_eigenvalues), np.imag(f_eigenvalues))
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title(r'Eigenvalues of $\nabla f$')

plt.figure()
plt.scatter(np.real(psi_eigenvalues), np.imag(psi_eigenvalues), label=r'Eigenvalues of $\nabla \psi$')
plt.scatter(np.real(approx_psi_eigenvalues), np.imag(approx_psi_eigenvalues), edgecolors='tab:orange', facecolor='none', label=r'$1- \exp(T \sigma)$')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.legend()

# Plot the leading eigenfunctions
plt.figure()
for n in range(10):
    plt.plot(x_array, psi_eigenvectors[:,n], label=f'Eigenvector {n}')
plt.xlabel(r'$x$')
plt.legend()

plt.show()