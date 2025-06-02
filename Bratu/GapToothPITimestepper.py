import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

import RBF

from Bratu.old.GapToothTimestepper import toNumpyArray, toPatch, euler_patch

def eulerPatchPITimestepper(u, dx, dt, Dt, K, T_patch, a, b, patch, n_teeth, params):
    N_PI_Steps = int(T_patch / Dt)
    for n in range(N_PI_Steps):
        # Do K Euler steps and record the last two solutions
        for _ in range(K-1):
            u = euler_patch(u, dx, dt, a, b, patch, n_teeth, params)
        u_final = euler_patch(u, dx, dt, a, b, patch, n_teeth, params) # Do one final step and return both
    
        # Record the local time derivative
        du_dt = (u_final - u) / dt

        # Extrapolate
        u = u_final + (Dt - K*dt) * du_dt

        # Verify boundary conditions
        if patch == 0:
            u[0] = 0.0
            u[-1] = u[-2] + b * dx
        elif patch == n_teeth - 1:
            u[0]  = u[1]  - a * dx
            u[-1] = 0.0
        else:
            u[0]  = u[1]  - a * dx
            u[-1] = u[-2] + b * dx

    # Return solution after T_patch time
    return u

def patchPIOneTimestep(u0, x_array, n_teeth, dx, dt, Dt, K, T_patch, params, solver='lu_direct', return_neumann=False):
   
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
        u_new = eulerPatchPITimestepper(u0[patch], dx, dt, Dt, K, T_patch, a, b, patch, n_teeth, params)

        return_u[patch] = u_new
        left_bcs[patch] = a
        right_bcs[patch] = b

    if return_neumann:
        return return_u, left_bcs, right_bcs
    return return_u

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
def psiPatch(u0, x_plot_array, n_teeth, dx, dt, Dt, K, T_patch, T_psi, params, verbose=False):
    if verbose:
        print('Evaluation ', psiPatch.count)

    u_sol = toPatch(x_plot_array, u0)
    for n in range(int(T_psi / T_patch)):
        u_sol = patchPIOneTimestep(u_sol, x_plot_array, n_teeth, dx, dt, Dt, K, T_patch, params, solver='lu_direct', return_neumann=False)

    return u0 - toNumpyArray(u_sol)

def gapToothProjectiveIntegrationEvolution():
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
    K = 2
    Dt = 4.e-6
    T = 0.5
    T_patch = 100 * dt
    for n in range(int(T / T_patch)):
        print('T =', (n+1) * T_patch)
        u_sol = patchPIOneTimestep(u_sol, x_plot_array, n_teeth, dx, dt, Dt, K, T_patch, params, solver='lu_direct', return_neumann=False)

    # Store the solution to file
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    filename = 'Evolution_PI_Steady_State_lambda=' + str(lam) + '.npy'
    np.save(directory + filename, u_sol)

    # Plot the solution of each tooth
    plt.plot(x_plot_array[0], u_sol[0], label=r'$u(x, t=$' + str(T) + r'$)$', color='blue')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], u_sol[i], color='blue')
    plt.xlabel(r'$x$')
    plt.title('Gap-Tooth with Projective Integration')
    plt.legend()
    plt.show()

def calculateSteadyState(_return=False):
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
    K = 2
    Dt = 4.e-6
    T_patch = 100 * dt
    T_psi = 1.e-2
    F = lambda u: psiPatch(u, x_plot_array, n_teeth, dx, dt, Dt, K, T_patch, T_psi, params, verbose=True)
    u_ss = opt.newton_krylov(F, u0, verbose=True, f_tol=1.e-14)
    if _return:
        return u_ss

    # Store the solution to file
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    filename = 'Newton-GMRES_PI_Steady_State_lambda=' + str(lam) + '.npy'
    np.save(directory + filename, u_ss)

    # Plot the solution of each tooth
    u_ss = toPatch(x_plot_array, u_ss)
    plt.plot(x_plot_array[0], u_ss[0], label='Newton-GMRES', color='blue')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], u_ss[i], color='blue')
    plt.xlabel(r'$x$')
    plt.title('Steady-State through Newton-Krylov')
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
    u_ss = calculateSteadyState(_return=True)

    # Eigenvalues through arnoldi
    dt = 1.e-6
    K = 2
    Dt = 4.e-6
    T_patch = 100 * dt
    T_psi = 1.e-2
    rdiff = 1.e-8
    psi_val =  psiPatch(u_ss, x_plot_array, n_teeth, dx, dt, Dt, K, T_patch, T_psi, params, verbose=True)
    print('psi_val', lg.norm(psi_val))
    M = n_teeth * n_points_per_tooth
    d_psi_mvp = lambda v: (psiPatch(u_ss + rdiff * v, x_plot_array, n_teeth, dx, dt, Dt, K, T_patch, T_psi, params, verbose=True) - psi_val) / rdiff
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
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    np.save(directory + 'GapToothPIEigenvalues_QR.npy', eigvals)
    np.save(directory + 'GapToothPIEigenvalues_Arnoldi.npy', eigvals_arnoldi)

    # Plot the eigenvalues
    plt.scatter(np.real(1-eigvals), np.imag(eigvals), alpha=0.5, label='QR Method')
    plt.scatter(np.real(1-eigvals_arnoldi), np.imag(eigvals_arnoldi), alpha=0.5, label='Arnoldi Method')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Eigenvalues of the Jacobian in Steady-State')
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
        gapToothProjectiveIntegrationEvolution()
    elif args.experiment == 'steady-state':
        calculateSteadyState()
    elif args.experiment == 'arnoldi':
        calculateEigenvalues()