import numpy as np
import numpy.linalg as lg
import numpy.random as rd
from Bratu.old.GapToothTimestepper import patchTimestepper, toPatch, toNumpyArray

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
store_directory = '/Users/hannesvdc/Research/Projects/Time-stepper/Bratu/data/'

# Run each initial condition
dt = 1.e-5
T_patch = 10 * dt
T = 1.0
n_initials = 100
rng = rd.RandomState()
for n in range(n_initials):
    print('Initial', n)
    k = rng.randint(1, 10)
    eps = rng.uniform(0.0, 0.01)
    perturbation = eps * np.sin(k * np.pi * x_array)
    perturbation_patch = []
    for i in range(n_teeth):
        perturbation_patch.append(perturbation[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
    u0 = toPatch(x_plot_array, u_ss + toNumpyArray(perturbation_patch))

    # Evolve
    _, evolution = patchTimestepper(x_plot_array, u0, dx, dt, T_patch, T, params, storeEvolution=True)

    # Store each patch separately and ignore the first 5 steps
    for patch in range(n_teeth):
        np.save(store_directory + 'Initial='+str(n)+'_patch='+str(patch) + '.npy', evolution[patch][5:,:])