"""
Vectorised gap-tooth + projective integration (PI)
==================================================

* Patches stored in one array  U[k, j]     (k = tooth, j = micro index).
* Every micro Euler step is done on *all* patches simultaneously.
* After K micro steps the local time-derivative is estimated and the
  solution is projected forward by  (Dt − K·dt).

Author: Hannes Vandecasteele, aided by ChatGPT(o3)
"""

import numpy as np
import matplotlib.pyplot as plt
import RBF

# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------
def rhs_vectorised(U, dx, params):
    """Central Laplacian + λ·eᵘ on every patch cell (axis-1 = micro grid)."""
    U_left  = np.roll(U, -1, axis=1)
    U_right = np.roll(U,  1, axis=1)
    U_xx    = (U_left - 2.0*U + U_right) / dx**2
    return U_xx + params['lambda'] * np.exp(U)


def euler_step(U, dx, dt, left_slope, right_slope, params):
    """One forward-Euler micro step *with* Dirichlet/Neumann BCs."""
    U_new = U + dt * rhs_vectorised(U, dx, params)

    # Enforce BCs (Dirichlet at outer edge, Neumann at internal edges)
    n_teeth = U_new.shape[0]
    U_new[0, 0]       = 0.0          # left end of global domain
    U_new[-1, -1]     = 0.0          # right end of global domain
    if n_teeth > 1:                  # interior edges
        U_new[1:,  0]  = U_new[1:, 1]   - left_slope[1:]  * dx
        U_new[:-1, -1] = U_new[:-1,-2]  + right_slope[:-1]* dx
    return U_new


def projective_microcycle(U, dx, dt, Dt, K, left_slope, right_slope, params):
    """
    One PI cycle of length Dt on **all** patches:
        • K Euler steps (store last two)
        • du/dt = (u_K − u_{K-1}) / dt
        • project: u ← u_K + (Dt − K·dt)·du/dt
        • re-impose BCs with same slopes (cheap)
    """
    for m in range(K-1):
        U = euler_step(U, dx, dt, left_slope, right_slope, params)

    U_prev = U.copy()
    U      = euler_step(U, dx, dt, left_slope, right_slope, params)

    du_dt  = (U - U_prev) / dt
    U      = U + (Dt - K*dt) * du_dt

    # BCs again after projection
    n_teeth = U.shape[0]
    U[0, 0]   = 0.0
    U[-1,-1]  = 0.0
    if n_teeth > 1:
        U[1:,  0]  = U[1:, 1]  - left_slope[1:]  * dx
        U[:-1,-1]  = U[:-1,-2] + right_slope[:-1]* dx
    return U
# ---------------------------------------------------------------------------
# Helpers: spline & slopes
# ---------------------------------------------------------------------------
def spline_slopes(U, x_array, solver="lu_direct"):
    """Return outward slopes aₖ, bₖ at every patch end-point."""
    n_teeth   = len(x_array)
    x_end     = np.empty(2*n_teeth)
    u_end     = np.empty_like(x_end)
    for k in range(n_teeth):
        x_end[2*k:2*k+2] = (x_array[k][0],  x_array[k][-1])
        u_end[2*k:2*k+2] = (U[k,0],          U[k,-1])
    spline     = RBF.RBFInterpolator(x_end, u_end, solver=solver)

    left  = np.fromiter((spline.derivative(x_array[k][0])   for k in range(n_teeth)),
                        dtype=float, count=n_teeth)
    right = np.fromiter((spline.derivative(x_array[k][-1])  for k in range(n_teeth)),
                        dtype=float, count=n_teeth)
    return left, right
# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------
def gaptooth_PI_vectorised(u0_patch, x_array,
                           dx, dt, Dt, K, T_patch, T,
                           params,
                           solver="lu_direct",
                           verbose=False):
    """
    Vectorised replacement for your `patchPIOneTimestep` loop.

    Parameters
    ----------
    u0_patch : list[ndarray]          – initial micro solution
    x_array  : list[ndarray]          – grids (same length each patch)
    dx, dt   : floats                 – micro spacing & micro step
    Dt       : float                  – PI coarse step (Dt ≥ K·dt)
    K        : int                    – # micro Euler steps used in PI
    T_patch  : float                  – horizon between spline rebuilds
    T        : float                  – final macro time
    params   : dict                   – {'lambda': …}
    """
    U = np.stack(u0_patch, axis=0)       # (n_teeth, n_micro)
    n_teeth, n_micro = U.shape

    n_patch_steps = int(np.round(T / T_patch))
    n_PI_steps    = int(np.round(T_patch / Dt))

    for p in range(1, n_patch_steps+1):
        if verbose:
            print(f"T = {p*T_patch:.4f}")

        # Build fresh spline from current end-points
        left_slope, right_slope = spline_slopes(U, x_array, solver=solver)

        # --- integrate over T_patch via successive PI cycles --------------
        for _ in range(n_PI_steps):
            U = projective_microcycle(U, dx, dt, Dt, K,
                                      left_slope, right_slope, params)

    # Return list-of-arrays to stay API-compatible
    return [U[k].copy() for k in range(n_teeth)]

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
    u0_patch = []
    for i in range(n_teeth):
        u0_patch.append(u0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
    
    # Time-stepping
    dt = 1.e-6
    K = 2
    Dt = 4.e-6
    T = 0.5
    T_patch = 100 * dt
    u_sol = gaptooth_PI_vectorised(u0_patch, x_plot_array, dx, dt, Dt, K, T_patch, T, params, verbose=True)

    # Load the non-vectorized time-evolution solution for verification
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    filename = 'Evolution_PI_Steady_State_lambda=' + str(lam) + '.npy'
    gt_pi_evolution = np.load(directory + filename)

    # Plot the solution of each tooth
    plt.plot(x_plot_array[0], gt_pi_evolution[0,:], label=r'$u(x, t=$' + str(T) + r'$)$ non-Vectorized', color='tab:orange')
    plt.plot(x_plot_array[0], u_sol[0], linestyle='--', label=r'$u(x, t=$' + str(T) + r'$)$ Vectorized', color='blue')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], gt_pi_evolution[i,:], color='tab:orange')
        plt.plot(x_plot_array[i], u_sol[i], linestyle='--', color='blue')
    plt.xlabel(r'$x$')
    plt.title('Gap-Tooth with Projective Integration')
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
        pass
    elif args.experiment == 'arnoldi':
        pass
    else:
        print('This experiment is not supported.')