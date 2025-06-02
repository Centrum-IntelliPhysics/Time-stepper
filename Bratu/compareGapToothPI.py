import numpy as np
import matplotlib.pyplot as plt

from Bratu.old.GapToothTimestepper import toPatch

# Model parameters
lam = 1.0

# Geometry parameters
n_teeth = 21
n_gaps = n_teeth - 1
gap_over_tooth_size_ratio = 1
n_points_per_tooth = 15
n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
dx = 1.0 / (N - 1)
x_array = np.linspace(0.0, 1.0, N)
x_plot_array = []
for i in range(n_teeth):
    x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])

# Compare all steady states (evolution and Newton-Krylov, Gap-Tooth and Gap-Tooth PI)
directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
gt_evolution = np.load(directory + 'Evolution_Steady_State_lambda=' + str(lam) + '.npy')
gt_nk = np.array(toPatch(x_plot_array, np.load(directory + 'Newton-GMRES_Steady_State_lambda=' + str(lam) + '.npy')))
gt_pi_evolution = np.load(directory + 'Evolution_PI_Steady_State_lambda=' + str(lam) + '.npy')
gt_pi_nk = np.array(toPatch(x_plot_array, np.load(directory + 'Newton-GMRES_PI_Steady_State_lambda=' + str(lam) + '.npy')))

# Plot all steady-states
plt.plot(x_plot_array[0], gt_evolution[0,:], label='Gap-Tooth Time- volution', color='tab:blue')
plt.plot(x_plot_array[0], gt_nk[0,:], label='Gap-Tooth Newton-Krylov', color='tab:orange')
plt.plot(x_plot_array[0], gt_pi_evolution[0,:], label='Gap-Tooth with PI Time Evolution', color='tab:red')
plt.plot(x_plot_array[0], gt_pi_nk[0,:], label='Gap-Tooth with PI Newton-Krylov', color='tab:green')
for i in range(1, n_teeth):
    plt.plot(x_plot_array[i], gt_evolution[i,:], color='tab:blue')
    plt.plot(x_plot_array[i], gt_nk[i,:], color='tab:orange')
    plt.plot(x_plot_array[i], gt_pi_evolution[i,:], color='tab:red')
    plt.plot(x_plot_array[i], gt_pi_nk[i,:], color='tab:green')
plt.xlabel(r'$x$')
plt.title('Steady-State Profile of the Bratu Equation')
plt.legend()

# Plot all eigenvalues
jitter = 0.01
gt_arnoldi = np.load(directory + 'GapToothEigenvalues_Arnoldi.npy')
gt_pi_arnoldi = np.load(directory + 'GapToothPIEigenvalues_Arnoldi.npy')
euler_arnoldi = np.load(directory + 'EulerEigenvalues_Arnoldi.npy')
analytic = np.load(directory + 'Analytic_Eigenvalues.npy')
plt.figure()
plt.scatter(np.real(1-gt_pi_arnoldi), jitter + np.imag(gt_pi_arnoldi), alpha=0.6, color='limegreen', label='Gap-Tooth with PI')
plt.scatter(np.real(1-gt_arnoldi), np.imag(gt_arnoldi), alpha=0.6, facecolors='none', edgecolors='tab:orange', label='Gap-Tooth')
plt.scatter(np.real(analytic), -jitter + np.imag(analytic), alpha=0.6, marker='x', color='k', label='Ground Truth')
plt.scatter(np.real(1-euler_arnoldi), 2*jitter + np.imag(euler_arnoldi), alpha=0.6, marker='^', color='steelblue', label='Euler')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.ylim((-0.5, 0.5))
plt.title('Leading Eigenvalues of the Jacobian in Steady State')
plt.legend()

plt.show()