import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

import RBF

def toPatch(x_plot_array, u):
    length = len(x_plot_array[0])
    u_patch = []
    for i in range(len(x_plot_array)):
        u_patch.append(u[i * length:(i+1)*length])
    return u_patch

def toNumpyArray(u_patch):
    length = u_patch[0].size
    u = np.zeros(len(u_patch) * length)
    for i in range(len(u_patch)):
        u[i * length:(i+1)*length] = u_patch[i]
    return u

# The PDE right-hand side without boundary conditions
def rhs(u, dx, params):
    u_left = np.roll(u, -1)
    u_right = np.roll(u, 1)
    u_xx = (u_left - 2.0*u + u_right) / dx**2

    u_rhs = u_xx + params['lambda'] * np.exp(u)
    return u_rhs

def euler_patch(u, dx, dt, a, b, patch, n_teeth, params):
    u_rhs = rhs(u, dx, params)
    u_new = u + dt * u_rhs

    # Apply the Dirichlet and Neumann boundary conditions
    if patch == 0:
        u_new[0] = 0.0
        u_new[-1] = u_new[-2] + b * dx
    elif patch == n_teeth - 1:
        u_new[0]  = u_new[1]  - a * dx
        u_new[-1] = 0.0
    else:
        u_new[0]  = u_new[1]  - a * dx
        u_new[-1] = u_new[-2] + b * dx
    return u_new

def eulerNeumannPatchTimestepper(u, dx, dt, T, a, b, patch, n_teeth, params):
    N_steps = int(T / dt)
    for _ in range(N_steps):
        u = euler_patch(u, dx, dt, a, b, patch, n_teeth, params)
    return u

def patchOneTimestep(u0, x_array, n_teeth, dx, dt, T_patch, params, solver='lu_direct'):
   
    # Build the interpolating spline based on left- and right endpoints
    x_spline_values = []
    u_spline_values = []
    for patch in range(n_teeth):
        x_spline_values.extend([x_array[patch][0], x_array[patch][-1]])
        u_spline_values.extend([u0[patch][0], u0[patch][-1]])
    u_spline = RBF.RBFInterpolator(x_spline_values, u_spline_values, solver=solver)

    # x_plot_array = np.linspace(0.0, 1.0, 1001)
    # plt.plot(x_plot_array, u_spline(x_plot_array), color='tab:green', linestyle='dashed', label='Radial Basis Interpolation')
    # for patch in range(len(u0)):
    #     if patch == 0:
    #         plt.plot(x_array[patch], u0[patch], color='tab:blue', label=r'$u(x)$ Patches')
    #     else:
    #         plt.plot(x_array[patch], u0[patch], color='tab:blue')
    # plt.xlabel(r'$x$')
    # plt.legend()
    # plt.show()

    # For each tooth: calculate Neumann boundary conditions and simulate in that tooth
    return_u = []
    for patch in range(n_teeth):
        left_x = x_array[patch][0]
        right_x = x_array[patch][-1]
        a = u_spline.derivative(left_x)
        b = u_spline.derivative(right_x)

        u_new = eulerNeumannPatchTimestepper(u0[patch], dx, dt, T_patch, a, b, patch, n_teeth, params)
        return_u.append(u_new)

    return return_u

def patchTimestepper(x_plot_array, u_sol, dx, dt, T_patch, T, params, verbose=False, storeEvolution=False):
    n_teeth = len(x_plot_array)
    n_patch_steps = int(T / T_patch)

    if storeEvolution:
        evolution = np.zeros((n_patch_steps+1, n_teeth * x_plot_array[0].size))
        evolution[0,:] = toNumpyArray(u_sol)

    for k in range(n_patch_steps):
        if verbose and k % 1000 == 0:
            print('t =', round(k*T_patch, 4))
        u_sol = patchOneTimestep(u_sol, x_plot_array, n_teeth, dx, dt, T_patch, params, solver='lu_direct')
        if storeEvolution:
            evolution[k+1,:] = toNumpyArray(u_sol)
    
    if storeEvolution:
        return u_sol, evolution
    return u_sol

def eval_counter(func):
    count = 0
    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        wrapper.count = count
        return func(*args, **kwargs)
    wrapper.count = count
    return wrapper

# Input u0 is a numpy array
@eval_counter
def psiPatch(x_plot_array, u0, dx, dt, T_patch, T, params):
    print('Evaluation ', psiPatch.count)
    u_sol = toPatch(x_plot_array, u0)
    u_new = patchTimestepper(x_plot_array, u_sol, dx, dt, T_patch, T, params)

    return u0 - toNumpyArray(u_new)

def gapToothEvolution():
    RBF.RBFInterpolator.lu_exists = False

    # Domain parameters
    n_teeth = 10
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 11
    n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
    N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
    dx = 1.0 / (N - 1)

    # Model parameters
    lam = 1.0
    params = {'lambda': lam}

    # Initial condition - Convert it to the Gap-Tooth datastructure
    x_array = np.linspace(0.0, 1.0, N)
    x_plot_array = []
    u0 = 0.0 * x_array
    u_sol = []
    for i in range(n_teeth):
        u_sol.append(u0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
    
    # Time-stepping
    dt = 1.e-5
    T = 100.0
    T_patch = 10 * dt
    u_sol = patchTimestepper(x_plot_array, u_sol, dx, dt, T_patch, T, params, verbose=True)

    # Store the solution to file
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    filename = 'Evolution_Steady_State_lambda=' + str(lam) + '.npy'
    np.save(directory + filename, u_sol)

    # Plot the solution of each tooth
    plt.plot(x_plot_array[0], u_sol[0], label=r'$u(x, t=$' + str(T) + r'$)$', color='blue')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], u_sol[i], color='blue')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()

def calculateSteadyState():
    RBF.RBFInterpolator.lu_exists = False

    # Domain parameters
    n_teeth = 10
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 11
    n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
    N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
    dx = 1.0 / (N - 1)

    # Model parameters
    lam = 1.0
    params = {'lambda': lam}

    # Initial condition - Convert it to the Gap-Tooth datastructure
    x_array = np.linspace(0.0, 1.0, N)
    x_plot_array = []
    u0 = 0.0 * x_array
    u_patch = []
    for i in range(n_teeth):
        u_patch.append(u0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
    u0 = toNumpyArray(u_patch)

    # Newton-GMRES
    dt = 1.e-5
    T_patch = 10 * dt
    T_psi = 1.0
    F = lambda u: psiPatch(x_plot_array, u, dx, dt, T_patch, T_psi, params)
    u_ss = opt.newton_krylov(F, u0, verbose=True, f_tol=1.e-14)

    # Store the solution to file
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    filename = 'Newton-GMRES_Steady_State_lambda=' + str(lam) + '.npy'
    np.save(directory + filename, u_ss)

    # Plot the solution of each tooth
    u_ss = toPatch(x_plot_array, u_ss)
    plt.plot(x_plot_array[0], u_ss[0], label='Newton-GMRES', color='blue')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], u_ss[i], color='blue')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()

def compareEvolutionandNewtonGMRES():
    # Domain parameters
    n_teeth = 10
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 11
    n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
    N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap

    # Initial condition - Convert it to the Gap-Tooth datastructure
    x_array = np.linspace(0.0, 1.0, N)
    x_plot_array = []
    for i in range(n_teeth):
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])

    # Load the data
    lam = 1.0
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    evolution = np.load(directory + 'Evolution_Steady_State_lambda=' + str(lam) + '.npy')
    ss = np.load(directory + 'Newton-GMRES_Steady_State_lambda=' + str(lam) + '.npy')
    print('evolution', evolution)
    print('ss', ss)
    ss = toPatch(x_plot_array, ss)
    print(ss)

    # Plot Both
    plt.plot(x_plot_array[0], evolution[0], label='Time Evolution', color='tab:blue')
    plt.plot(x_plot_array[0], ss[0], label='Newton-GMRES', color='tab:orange', linestyle='dashed')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], evolution[i], color='tab:blue')
        plt.plot(x_plot_array[i], ss[i], color='tab:orange', linestyle='dashed')
    plt.xlabel(r'$x$')
    plt.title('Steady-State of the Bratu Equation via the Gap-Tooth Scheme')
    plt.legend()
    plt.show()

def calculateEigenvalues():
    RBF.RBFInterpolator.lu_exists = False

    # Domain parameters
    n_teeth = 10
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 11
    n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
    N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
    dx = 1.0 / (N - 1)

    # Model parameters
    lam = 1.0
    params = {'lambda': lam}

    # Load the steady-state
    x_array = np.linspace(0.0, 1.0, N)
    x_plot_array = []
    for i in range(n_teeth):
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    u_ss = np.load(directory + 'Newton-GMRES_Steady_State_lambda=' + str(lam) + '.npy')

    # Eigenvalues through arnoldi
    dt = 1.e-5
    T_patch = 10 * dt
    T_psi = 1.0
    rdiff = 1.e-8
    psi_val = psiPatch(x_plot_array, u_ss, dx, dt, T_patch, T_psi, params)
    print(lg.norm(psi_val))
    M = n_teeth * n_points_per_tooth
    d_psi_mvp = lambda v: (psiPatch(x_plot_array, u_ss + rdiff * v, dx, dt, T_patch, T_psi, params) - psi_val) / rdiff
    Dpsi = slg.LinearOperator(shape=(M,M), matvec=d_psi_mvp)

    # Build the full Jacobian matrix
    Dpsi_matrix = np.zeros((M,M))
    for i in range(M):
        Dpsi_matrix[:,i] = Dpsi.matvec(np.eye(M)[:,i])
    eigvals, eigvecs = lg.eig(Dpsi_matrix)

    # Store the eigenvalues
    np.save(directory + 'GapToothEigenvalues.npy', eigvals)

    # Plot the eigenvalues
    plt.scatter(np.real(eigvals), np.imag(eigvals), label='Gap-Tooth Timestepper')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    calculateEigenvalues()