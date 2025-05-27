import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

import RBF

from GapToothTimestepper import toNumpyArray, toPatch, euler_patch

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
    Dt = 2.e-6
    T = 0.5
    T_patch = 100 * dt
    for n in range(int(T / T_patch)):
        u_sol = patchPIOneTimestep(u_sol, x_plot_array, n_teeth, dx, dt, Dt, K, T_patch, params, solver='lu_direct', return_neumann=False)

    # Store the solution to file
    #directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    #filename = 'Evolution_Steady_State_lambda=' + str(lam) + '.npy'
    #np.save(directory + filename, u_sol)

    # Plot the solution of each tooth
    plt.plot(x_plot_array[0], u_sol[0], label=r'$u(x, t=$' + str(T) + r'$)$', color='blue')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], u_sol[i], color='blue')
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