import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt
import RBF
from Bratu.old.GapToothTimestepper import psiPatch

# Domain parameters
n_teeth = 21
n_gaps = n_teeth - 1
gap_over_tooth_size_ratio = 1
n_points_per_tooth = 15
n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
N = n_teeth * n_points_per_tooth
M = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
dx = 1.0 / (M - 1)
x_array = np.linspace(0.0, 1.0, M)
x_plot_array = []
for i in range(n_teeth):
    x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])

dt = 1.e-6
T_patch = 10 * dt
T_psi = 1.e-4
rdiff = 1.e-8
directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
params = {}

def G(x, lam):
    params['lambda'] = lam
    return psiPatch(x_plot_array, x, dx, dt, T_patch, T_psi, params)

def dGdx_v(x, v, lam):
    return (G(x + rdiff * v, lam) - G(x, lam)) / rdiff

def dGdlam(x, lam):
    return (G(x, lam + rdiff) - G(x, lam)) / rdiff

# Calculates the tangent to the path at the current point as (Gx^{-1} G_lam, -1).
def computeTangent(Gx_v, G_lam, prev_tangent, tolerance):
    x0 = prev_tangent[0:N] / prev_tangent[N]
    A = slg.LinearOperator(matvec=Gx_v, shape=(N,N))

    _tangent = slg.gmres(A, G_lam, x0=x0, atol=tolerance)[0]
    tangent = np.append(_tangent, -1.0)
    tangent = tangent / lg.norm(tangent)

    if np.dot(tangent, prev_tangent) > 0:
        return tangent
    else:
        return -tangent

"""
The Internal Numerical Continuation Routine.
"""
def numericalContinuation(x0, lam0, initial_tangent, max_steps, ds, ds_min, ds_max, tolerance):
    x = np.copy(x0)
    lam = lam0
    prev_tangent = np.copy(initial_tangent)

    x_path = [np.copy(x)]
    lam_path = [lam]
    print('Step {0:3d}:\t |u|_inf: {1:4f}\t lambda: {2:4f}\t ds: {3:6f}'.format(0, lg.norm(x_path[0], ord=np.inf), lam, ds))

    for n in range(1, max_steps+1):
        if lam < 0.1:
            print('Artificial end point reached, quiting this branch.')
            break

		# Calculate the tangent to the curve at current point 
        Gx_v = lambda v: dGdx_v(x, v, lam)
        Glam = dGdlam(x, lam)
        tangent = computeTangent(Gx_v, Glam, prev_tangent, tolerance)

		# Create the extended system for corrector: z = (u, lam)
        N_opt = lambda z: np.dot(tangent, z - np.append(x, lam)) + ds
        F = lambda z: np.append(G(z[0:N], z[N]), N_opt(z))

		# Our implementation uses adaptive timetepping
        while ds > ds_min:
			# Predictor: Extrapolation
            x_p = x + ds * tangent[0:N]
            lam_p = lam + ds * tangent[N]
            z_p = np.append(x_p, lam_p)

			# Corrector: Newton - Krylov
            try:
                z_new = opt.newton_krylov(F, z_p, f_tol=tolerance)
                x = z_new[0:N]
                lam = z_new[N]
                x_path.append(np.copy(x))
                lam_path.append(lam)

				# Updating the arclength step and tangent vector
                ds = min(1.2*ds, ds_max)
                prev_tangent = np.copy(tangent)

                break
            except:
                # Decrease arclength if the corrector fails.
                ds = max(0.5*ds, ds_min)
        else:
            print('Minimal Arclength Size is too large. Aborting.')
            return x_path, lam_path
		
        print('Step {0:3d}:\t |u|_inf: {1:4f}\t lambda: {2:4f}\t ds: {3:6f}'.format(n, lg.norm(x_path[-1], ord=np.inf), lam, ds))

    return x_path, lam_path

def computeBifurcationDiagram():
    RBF.RBFInterpolator.lu_exists = False
    
    # Model parameters
    lam0 = 1.0
    params['lambda'] = lam0

    # Continuation Parameters
    tolerance = 1.e-10
    max_steps = 3000
    ds_min = 1.e-6
    ds_max = 0.01
    ds = 0.001

    # Initial condition - Convert it to the Gap-Tooth datastructure
    print('Loading the initial condition...')
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/Bratu/'
    filename = 'Newton-GMRES_Steady_State_lambda=' + str(lam0) + '.npy'
    u0 = np.load(directory + filename)

    # Calculate the tangent to the path at the initial condition x0
    print('Computing the initial tangent vector...')
    rng = rd.RandomState()
    random_tangent = rng.normal(0.0, 1.0, N+1)
    initial_tangent = computeTangent(lambda v: dGdx_v(u0, v, lam0), dGdlam(u0, lam0), random_tangent / lg.norm(random_tangent), tolerance)
    initial_tangent = initial_tangent / lg.norm(initial_tangent)

    # Do actual numerical continuation in both directions
    print(initial_tangent[-1])
    x1_path, lam1_path = numericalContinuation(u0, lam0,  initial_tangent, max_steps, ds, ds_min, ds_max, tolerance)
    x2_path, lam2_path = numericalContinuation(u0, lam0, -initial_tangent, max_steps, ds, ds_min, ds_max, tolerance)

    # Plot both branches
    x1_path = np.array(x1_path)
    x2_path = np.array(x2_path)
    lam1_path = np.array(lam1_path)
    lam2_path = np.array(lam2_path)
    plot_x1_path = np.max(np.abs(x1_path[:, 0:N]), axis=1)
    plot_x2_path = np.max(np.abs(x2_path[:, 0:N]), axis=1)
    plt.plot(lam1_path, plot_x1_path, color='blue')
    plt.plot(lam2_path, plot_x2_path, color='blue')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$|u|_{\infty}$')
    plt.show()

if __name__ == '__main__':
    computeBifurcationDiagram()