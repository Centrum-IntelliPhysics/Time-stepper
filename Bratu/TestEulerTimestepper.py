import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

from EulerTimestepper import timestepper, psi, calculateSteadyState, calculateRHSEigenvalues, calculateLeadingEigenvalues

def testEulerTimestepper():
    # Model Parameters
    lam = 1.0
    params = {'lambda': lam}

    # Geometry Parameters
    N = 25
    x_array = np.linspace(0.0, 1.0, N)
    dx = 1.0 / N
    u0 = 0.0 * x_array

    # Time-stepping parameters
    T = 100.0
    dt = 1.e-4
    T_psi = 1.0

    # Do time-evolution
    print('Time Stepping...')
    u_evol = timestepper(u0, dx, dt, T, params, verbose=False)
    print('Psi at T =', T, ' :', lg.norm(psi(u_evol, T_psi, dx, dt, params)))

    # Do Newton-GMRES
    print('Newton-GMRES')
    u_ss = calculateSteadyState(u0, T_psi, dx, dt, params, verbose=True)
    print('Psi in Steady State :', lg.norm(psi(u_ss, T_psi, dx, dt, params)))

    # Calculate the Eigenvalues using the Arnoldi method. First of the rhs, then of the timestepper
    k = N - 2
    f_eigvals, f_eigvecs = calculateRHSEigenvalues(u_ss, dx, params, k=k)
    psi_eigvals, psi_eigvecs = calculateLeadingEigenvalues(u_ss, T_psi, dx, dt, params, k=k)
    approx_psi_eigvals = 1.0 - np.exp(T_psi * f_eigvals)

    # Plot the result
    plt.plot(x_array, u_evol, label=r'Solution at $T =$' + str(T))
    plt.plot(x_array, u_ss, label='Newton-GMRES Steady State')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u(x)$')
    plt.legend()

    plt.figure()
    plt.scatter(np.real(f_eigvals), np.imag(f_eigvals), label='PDE')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('PDE Eigenvalues')
    plt.legend()
    plt.figure()
    plt.scatter(np.real(psi_eigvals), np.imag(psi_eigvals), label='Time Stepper')
    plt.scatter(np.real(approx_psi_eigvals), np.imag(approx_psi_eigvals), facecolors='none', edgecolors='tab:orange', label='Analytic Correspondence')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Time Stepper Eigenvalues')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    testEulerTimestepper()