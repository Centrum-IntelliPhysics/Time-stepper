import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

import RBF

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

def patchTimestepper():
    RBF.RBFInterpolator.lu_exists = False

    # Domain parameters
    n_teeth = 10
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 11
    n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
    N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
    dx = 1.0 / (N - 1)
    print('N =', N, 'dx =', dx)

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
    n_patch_steps = int(T / T_patch)
    for k in range(n_patch_steps):
        if k % 1000 == 0:
            print('t =', round(k*T_patch, 4))
        u_sol = patchOneTimestep(u_sol, x_plot_array, n_teeth, dx, dt, T_patch, params, solver='lu_direct')

    # Plot the solution of each tooth
    plt.plot(x_plot_array[0], u_sol[0], label=r'$u(x, t=$' + str(T) + r'$)$', color='blue')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], u_sol[i], color='blue')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    patchTimestepper()