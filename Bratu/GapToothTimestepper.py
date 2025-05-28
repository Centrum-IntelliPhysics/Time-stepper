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

def patchOneTimestep(u0, x_array, n_teeth, dx, dt, T_patch, params, solver='lu_direct', return_neumann=False):
   
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

    # Function to process each patch
    return_u = [None] * n_teeth
    left_bcs = [None] * n_teeth
    right_bcs = [None] * n_teeth
    for patch in range(n_teeth):
        left_x = x_array[patch][0]
        right_x = x_array[patch][-1]
        a = u_spline.derivative(left_x)
        b = u_spline.derivative(right_x)

        # Calculate the new state using the timestepper
        u_new = eulerNeumannPatchTimestepper(u0[patch], dx, dt, T_patch, a, b, patch, n_teeth, params)

        return_u[patch] = u_new
        left_bcs[patch] = a
        right_bcs[patch] = b

    if return_neumann:
        return return_u, left_bcs, right_bcs
    return return_u

def patchTimestepper(x_plot_array, u_sol, dx, dt, T_patch, T, params, verbose=False, storeEvolution=False):
    n_teeth = len(x_plot_array)
    n_micro_points = x_plot_array[0].size
    n_patch_steps = int(T / T_patch)

    # Evolution is a datastructure for the DeepONet. It is a list of n_teeth patches, and each patch is a 
    # numpy array with 2*n_micro_points + 2 elements to store (u[patch], left_bc, right_bc, x[patch])
    if storeEvolution:
        evolution = n_teeth * [np.zeros((n_patch_steps+1, n_micro_points + 2 + n_micro_points))]
        for patch in range(n_teeth):
            evolution[patch][0,0:n_micro_points] = u_sol[patch]
            evolution[patch][0,n_micro_points] = 0.0
            evolution[patch][0,n_micro_points+1] = 0.0
            evolution[patch][0,n_micro_points+2:] = x_plot_array[patch]

    for k in range(n_patch_steps):
        if verbose and k % 1000 == 0:
            print('t =', round(k*T_patch, 4))

        if storeEvolution:
            u_sol, left_bcs, right_bcs = patchOneTimestep(u_sol, x_plot_array, n_teeth, dx, dt, T_patch, params, solver='lu_direct', return_neumann=True)
            for patch in range(n_teeth):
                evolution[patch][k+1,0:n_micro_points] = u_sol[patch]
                evolution[patch][k+1,n_micro_points] = left_bcs[patch]
                evolution[patch][k+1,n_micro_points+1] = right_bcs[patch]
                evolution[patch][k+1,n_micro_points+2:] = x_plot_array[patch]
        else:
            u_sol = patchOneTimestep(u_sol, x_plot_array, n_teeth, dx, dt, T_patch, params, solver='lu_direct')
            
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
def psiPatch(x_plot_array, u0, dx, dt, T_patch, T_psi, params, verbose=False):
    if verbose:
        print('Evaluation ', psiPatch.count)
    u_sol = toPatch(x_plot_array, u0)
    u_new = patchTimestepper(x_plot_array, u_sol, dx, dt, T_patch, T_psi, params, verbose=verbose)

    return u0 - toNumpyArray(u_new)

def gapToothEvolution():
    RBF.RBFInterpolator.lu_exists = False

    # Domain parameters
    n_teeth = 21
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 15
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
    dt = 1.e-6
    T = 0.5
    T_patch = 100 * dt
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
    n_teeth = 21
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 15
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
    dt = 1.e-6
    T_patch = 10 * dt
    T_psi = 1.e-3
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

def calculateEigenvalues():
    RBF.RBFInterpolator.lu_exists = False

    # Domain parameters
    n_teeth = 21
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 15
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
    dt = 1.e-6
    T_patch = 10 * dt
    T_psi = 1.e-4
    rdiff = 1.e-8
    psi_val = psiPatch(x_plot_array, u_ss, dx, dt, T_patch, T_psi, params)
    print(lg.norm(psi_val))
    M = n_teeth * n_points_per_tooth
    d_psi_mvp = lambda v: (psiPatch(x_plot_array, u_ss + rdiff * v, dx, dt, T_patch, T_psi, params, verbose=True) - psi_val) / rdiff
    Dpsi = slg.LinearOperator(shape=(M,M), matvec=d_psi_mvp)

    # Build the full Jacobian matrix
    Dpsi_matrix = np.zeros((M,M))
    for i in range(M):
        Dpsi_matrix[:,i] = Dpsi.matvec(np.eye(M)[:,i])
    eigvals, eigvecs = lg.eig(Dpsi_matrix)

    # Calculate the eigenvaleus using arnoldi
    print('Arnoldi Method')
    eigvals_arnoldi = slg.eigs(Dpsi, k=10, which='SM', return_eigenvectors=False)

    # Store the eigenvalues
    np.save(directory + 'GapToothEigenvalues_QR.npy', eigvals)
    np.save(directory + 'GapToothEigenvalues_Arnoldi.npy', eigvals_arnoldi)

    # Plot the eigenvalues
    plt.scatter(np.real(1-eigvals), np.imag(eigvals), alpha=0.5, label='QR Method')
    plt.scatter(np.real(1-eigvals_arnoldi), np.imag(eigvals_arnoldi), alpha=0.5, label='Arnoldi Method')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.show()

def parseArguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', nargs='?', dest='experiment')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    if args.experiment == 'evolution':
        gapToothEvolution()
    elif args.experiment == 'steady-state':
        calculateSteadyState()
    elif args.experiment == 'arnoldi':
        calculateEigenvalues()