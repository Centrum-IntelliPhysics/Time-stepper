import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

from EulerTimestepper import psi, calculateSteadyState

N = 50
x_array = np.linspace(0.0, 1.0, N)
dx = 1.0 / N
dt = 0.001
dt = 1.e-4
T_psi = 1.0
params = {}
rdiff = 1.e-8
directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'

def G(x, lam):
    params['lambda'] = lam
    return psi(x, T_psi, dx, dt, params)

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
    print_str = 'Step {0:3d}:\t |u|_inf: {1:4f}\t lambda: {2:4f}\t ds: {3:6f}'.format(0, np.max(np.abs(x_path[0][0:N])), lam, ds)
    print(print_str)

    for n in range(1, max_steps+1):
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
		
        print_str = 'Step {0:3d}:\t |u|_inf: {1:4f}\t lambda: {2:4f}\t ds: {3:6f}'.format(n, np.max(np.abs(x_path[-1][0:N])), lam, ds)
        print(print_str)

    return x_path, lam_path

"""
Routine that calculates the bifurcation diagram of a timestepper for the Fitzhugh-Nagumo PDE. Steady states of 
the pde equal fixex points of the timespper, or zeros of psi(x) = (x - s_T(x)) / T, with s_T the timestepper.
"""
def calculateBifurcationDiagram():
    lam0 = 1.0
    params['lambda'] = lam0
    u0 = calculateSteadyState(0.0*x_array, T_psi, dx, dt, params)

    # Continuation Parameters
    tolerance = 1.e-8
    max_steps = 150
    ds_min = 1.e-6
    ds_max = 0.001
    ds = 0.001

    # Calculate the tangent to the path at the initial condition x0
    rng = rd.RandomState()
    random_tangent = rng.normal(0.0, 1.0, N+1)
    initial_tangent = computeTangent(lambda v: dGdx_v(u0, v, lam0), dGdlam(u0, lam0), random_tangent / lg.norm(random_tangent), tolerance)
    initial_tangent = initial_tangent / lg.norm(initial_tangent)

    # Do actual numerical continuation in both directions
    if initial_tangent[-1] < 0.0: # Decreasing lam
        print('Increasing lam first')
        sign = 1.0
    else:
        sign = -1.0
    x1_path, lam1_path = numericalContinuation(u0, lam0,  sign * initial_tangent, max_steps, ds, ds_min, ds_max, tolerance)
    #x2_path, lam2_path = numericalContinuation(u0, lam0, -sign * initial_tangent, max_steps, ds, ds_min, ds_max, tolerance)

    # Store the full path
    x1_path = np.array(x1_path)
    #x2_path = np.array(x2_path)
    lam1_path = np.array(lam1_path)
    #lam2_path = np.array(lam2_path)
    #np.save(directory + 'euler_bf_diagram.npy', np.hstack((x1_path, lam1_path[:,np.newaxis], x2_path, lam2_path[:,np.newaxis])))

    # Plot both branches
    plot_x1_path = np.max(np.abs(x1_path[:, 0:N]), axis=1)
    #plot_x2_path = np.max(np.abs(x2_path[:, 0:N]), axis=1)
    plt.plot(lam1_path, plot_x1_path, color='blue')
    #plt.plot(lam2_path, plot_x2_path, color='blue')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$|u|_{\infty}$')
    plt.show()


if __name__ == '__main__':
    calculateBifurcationDiagram()