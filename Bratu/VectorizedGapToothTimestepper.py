"""
Vectorised gap-tooth patch time-stepper.
---------------------------------------

*  All patches are kept in a single array  U[k, j]
   (k = patch index, j = micro-grid index within the patch).

*  The micro-stepper integrates *all* patches simultaneously.
*  Neumann boundary slopes are refreshed once per macro step
   with a radial-basis spline fitted to the patch end-points.

Author: Hannes Vandecasteele, aided by ChatGPT(o3)
"""

import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt
import RBF

# ----------------------------------------------------------------------
# Low-level building blocks
# ----------------------------------------------------------------------
def rhs_vectorised(U, dx, params):
    """
    Second-order central Laplacian + stiff source  λ e^u  on **every**
    patch cell in one shot.

    Parameters
    ----------
    U : ndarray, shape (n_teeth, n_micro)
    dx : float
    params : dict  - needs key 'lambda'

    Returns
    -------
    ndarray, same shape as U
    """
    U_left  = np.roll(U, -1, axis=1)
    U_right = np.roll(U,  1, axis=1)
    U_xx    = (U_left - 2.0*U + U_right) / dx**2
    return U_xx + params['lambda'] * np.exp(U)


def euler_vectorised(U, dx, dt, left_slope, right_slope, params):
    """
    One forward-Euler micro step on **all patches**.

    Neumann/Dirichlet assignment mirrors your original `euler_patch`:

        patch 0 :   u(0) = 0          ,  u_x(L⁻) =  +b
        ...
        patch k :   u_x(0⁺) = -a      ,  u_x(L⁻) =  +b
        ...
        patch N :   u_x(0⁺) = -a      ,  u(L)   = 0
    """
    U_new = U + dt * rhs_vectorised(U, dx, params)

    n_teeth = U_new.shape[0]

    # Left boundary of global domain  (Dirichlet)
    U_new[0, 0] = 0.0
    # Right boundary of global domain (Dirichlet)
    U_new[-1, -1] = 0.0

    # Interior left edges  (Neumann, except patch 0)
    if n_teeth > 1:
        U_new[1:, 0] = U_new[1:, 1] - left_slope[1:] * dx
        # Interior right edges (Neumann, except last patch)
        U_new[:-1, -1] = U_new[:-1, -2] + right_slope[:-1] * dx

    # Patch 0, right edge  (Neumann with +b)  – already covered by slice above
    # Last patch, left edge (Neumann with –a) – already covered by slice above

    return U_new


# ----------------------------------------------------------------------
# Helpers for spline + slopes
# ----------------------------------------------------------------------
def build_spline_and_slopes(U, x_array, solver="lu_direct"):
    """
    Fit the radial-basis spline through the 2 * n_teeth end-points and
    return the outward normal slopes a_k , b_k used in the Neumann BC.

    Returns
    -------
    left_slope  : ndarray (n_teeth,)
    right_slope : ndarray (n_teeth,)
    """
    n_teeth = len(x_array)
    # Gather coordinates and values of all end-points
    x_end   = np.empty(2 * n_teeth)
    u_end   = np.empty_like(x_end)
    for k in range(n_teeth):
        x_end[2*k    ] = x_array[k][0]     # left end
        x_end[2*k + 1] = x_array[k][-1]    # right end
        u_end[2*k    ] = U[k, 0]
        u_end[2*k + 1] = U[k, -1]

    spline = RBF.RBFInterpolator(x_end, u_end, solver=solver)

    left_slope  = np.fromiter(
        (spline.derivative(x_array[k][0])   for k in range(n_teeth)),
        dtype=float, count=n_teeth
    )
    right_slope = np.fromiter(
        (spline.derivative(x_array[k][-1])  for k in range(n_teeth)),
        dtype=float, count=n_teeth
    )

    # Note:  in your notation  a = left_slope,  b = right_slope
    return left_slope, right_slope


# ----------------------------------------------------------------------
# Public driver --- reproduces patchTimestepper
# ----------------------------------------------------------------------
def gaptooth_vectorised(u0_patch, x_array, dx, dt, T_patch, T,
                        params, verbose=False):
    """
    Vectorised replacement for `patchTimestepper`.

    Parameters
    ----------
    u0_patch : list[ndarray]   - initial micro solution, one array per patch
    x_array  : list[ndarray]   - micro grids (same length across patches)
    dx, dt   : floats
    T_patch  : float           - micro horizon integrated between spline refreshes
    T        : float           - macro time to integrate to
    params   : dict            - PDE parameters (at least 'lambda')
    """
    U = np.stack(u0_patch, axis=0)      # (n_teeth, n_micro)
    n_teeth, n_micro = U.shape
    n_patch_steps    = int(np.round(T / T_patch))
    n_micro_steps    = int(np.round(T_patch / dt))

    for step in range(1, n_patch_steps+1):
        if verbose and step % 1000 == 0:
            print(f"t = {step*T_patch:.4f}")

        # ------------------------------------------------------------------
        # 1. recompute Neumann slopes with fresh spline
        # ------------------------------------------------------------------
        left_slope, right_slope = build_spline_and_slopes(U, x_array)

        # ------------------------------------------------------------------
        # 2. integrate all patches synchronously for T_patch using dt
        # ------------------------------------------------------------------
        for _ in range(n_micro_steps):
            U = euler_vectorised(U, dx, dt, left_slope, right_slope, params)

    return [U[k].copy() for k in range(n_teeth)]

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
def psiPatch(u0_numpy, x_plot_array, dx, dt, T_patch, T_psi, params, verbose=False):
    if verbose:
        print('Evaluation ', psiPatch.count)
    u0_patch = toPatch(x_plot_array, u0_numpy)
    u_new_patch = gaptooth_vectorised(u0_patch, x_plot_array, dx, dt, T_patch, T_psi, params, verbose=False)

    return u0_numpy - toNumpyArray(u_new_patch)

# ----------------------------------------------------------------------
# Convenience wrappers to convert between old and new layouts
# ----------------------------------------------------------------------
def list_to_array(u_patch):
    """ From list-of-1-D to 2-D array (n_teeth, n_micro) """
    return np.stack(u_patch, axis=0)

def array_to_list(U):
    """ From 2-D array back to list-of-1-D (to stay API-compatible) """
    return [U[k].copy() for k in range(U.shape[0])]

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
    u0_patch = []
    for i in range(n_teeth):
        u0_patch.append(u0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
    
    # Time-stepping
    dt = 1.e-6
    T = 0.5
    T_patch = 100 * dt
    u_sol = gaptooth_vectorised(u0_patch, x_plot_array, dx, dt, T_patch, T, params, verbose=True)

    # Load reference time-evolution of unvectorized code for checking correctness
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    gt_evolution = np.load(directory + 'Evolution_Steady_State_lambda=' + str(lam) + '.npy')

    # Plot the solution of each tooth
    plt.plot(x_plot_array[0], gt_evolution[0,:], label=r'$u(x, t=$' + str(T) + r'$)$ non-vectorized', color='tab:orange')
    plt.plot(x_plot_array[0], u_sol[0], label=r'$u(x, t=$' + str(T) + r'$)$ vectorized', linestyle='--', color='blue')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], gt_evolution[i,:]+0.0001, color='tab:orange')
        plt.plot(x_plot_array[i], u_sol[i], linestyle='--', color='blue')
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
    u0_patch = []
    for i in range(n_teeth):
        u0_patch.append(u0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
    u0_numpy = toNumpyArray(u0_patch)

    # Newton-GMRES
    dt = 1.e-6
    T_patch = 10 * dt
    T_psi = 1.e-2
    F = lambda u: psiPatch(u, x_plot_array, dx, dt, T_patch, T_psi, params)
    u_ss_numpy = opt.newton_krylov(F, u0_numpy, verbose=True, f_tol=1.e-14)

    # Load reference time-evolution of unvectorized code for checking correctness
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    gt_nk = np.array(toPatch(x_plot_array, np.load(directory + 'Newton-GMRES_Steady_State_lambda=' + str(lam) + '.npy')))

    # Plot the solution of each tooth
    u_ss_patch = toPatch(x_plot_array, u_ss_numpy)
    plt.plot(x_plot_array[0], gt_nk[0,:], label='Newton-GMRES non-vectorized', color='tab:orange')
    plt.plot(x_plot_array[0], u_ss_patch[0], label='Newton-GMRES Vectorized', linestyle='--', color='blue')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], gt_nk[i,:], color='tab:orange')
        plt.plot(x_plot_array[i], u_ss_patch[i], linestyle='--', color='blue')
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
    u_ss_numpy = np.load(directory + 'Newton-GMRES_Steady_State_lambda=' + str(lam) + '.npy')

    # Eigenvalues through arnoldi
    dt = 1.e-6
    T_patch = 10 * dt
    T_psi = 1.e-2
    rdiff = 1.e-8
    psi_val = psiPatch(u_ss_numpy, x_plot_array, dx, dt, T_patch, T_psi, params)
    M = n_teeth * n_points_per_tooth
    d_psi_mvp = lambda v: (psiPatch(u_ss_numpy + rdiff * v, x_plot_array, dx, dt, T_patch, T_psi, params, verbose=True) - psi_val) / rdiff
    Dpsi = slg.LinearOperator(shape=(M,M), matvec=d_psi_mvp)

    # Build the full Jacobian matrix
    print('QR Method')
    Dpsi_matrix = np.zeros((M,M))
    for i in range(M):
        Dpsi_matrix[:,i] = Dpsi.matvec(np.eye(M)[:,i])
    eigvals, eigvecs = lg.eig(Dpsi_matrix)

    # Calculate the eigenvaleus using arnoldi
    print('Arnoldi Method')
    eigvals_arnoldi = slg.eigs(Dpsi, k=10, which='SM', return_eigenvectors=False)

    # Load the non-vectorized eigenvalues for visual comparison and verification
    gt_arnoldi = np.load(directory + 'GapToothEigenvalues_Arnoldi.npy')

    # Plot the eigenvalues
    jitter = 0.001
    plt.scatter(np.real(1-eigvals), np.imag(eigvals), alpha=0.5, label='Vectorized QR Method')
    plt.scatter(np.real(1-eigvals_arnoldi), np.imag(eigvals_arnoldi) + jitter, alpha=0.5, label='Vectorized Arnoldi Method')
    plt.scatter(np.real(1-gt_arnoldi), np.imag(gt_arnoldi) - jitter, alpha=0.6, facecolors='none', edgecolors='tab:orange', label='Non-Vectorized Gap-Tooth')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.ylim((-0.4, 0.4))
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
    else:
        print('This experiment is not supported.')