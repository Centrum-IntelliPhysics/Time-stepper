import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

from EulerTimestepper import euler_timestepper, calculateSteadyState, psi

def testInvariantSolution():
    D = 0.1
    params = {'D': D}

    N = 50
    x_array = np.linspace(0.0, 1.0, N)
    dx = 1.0 / N
    u0 = 1.0 + np.sin(0.5 * np.pi * x_array) # Dirichlet boundary conditions u0(t,0)=1, u0(t,1)=2
    print('Minimal timestep for stability:', dx**2 / (2.0 * D))

    # Do time-evolution with this initial condition
    dt = 0.001
    T = 100.0
    u_final = euler_timestepper(u0, dx, dt, T, params)
    psi_u_final = psi(u_final, T, dx, dt, params)
    print('Psi after Time Evolution:', lg.norm(psi_u_final))

    # Calculate the steady state using newton-krylov
    T_psi = 1.0
    u_ss = calculateSteadyState(u0, T_psi, dx, dt, params)
    psi_u_ss = psi(u_ss, T_psi, dx, dt, params)
    print('Psi after Newton-GMRES:', lg.norm(psi_u_ss))

    # Plot both steady-states
    plt.plot(x_array, u0, label='Initial Condition')
    plt.plot(x_array, u_final, label='Time Evolution')
    plt.plot(x_array, u_ss, linestyle='dashed', label='Newton-GMRES')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u(x)$', rotation=0)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testInvariantSolution()