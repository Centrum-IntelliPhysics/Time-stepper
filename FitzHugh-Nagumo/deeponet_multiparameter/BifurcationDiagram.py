import torch as pt
import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt

from DeepONet import DeepONet

pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float64)

N = 200
L = 20.0
dt = 0.001
T = 1.0
grid_ext = pt.linspace(0.0, 1.0, N)[:,None]

p = 200
branch_layers = [400, 100, 100, 100, 100, 2*p]
trunk_layers = [2, 100, 100, 100, 100, 2*p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.load_state_dict(pt.load('./Results/model_deeponet_manyeps_fhn.pth', weights_only=True))

def deeponet(x, eps):
    input = pt.concatenate((pt.tile(x, dims=(N, 1)), grid_ext, eps * pt.ones((N,1))), dim=1)
    output = network.forward(input)
    return pt.concatenate((output[:,0], output[:,1]))

def psi(x0, eps, T_psi, dt):
    x = pt.from_numpy(x0)
    n = int(T_psi / dt)
    for _ in range(n):
        x = deeponet(x, eps)
    return x0 - x.numpy()

def G(x, eps):
    return psi(x, eps, T, dt)

def dGdx_v(x, v, eps):
    rdiff = 1.e-8
    return (G(x + rdiff * v, eps) - G(x, eps)) / rdiff

def dGdeps(x, eps):
    rdiff = 1.e-8
    return (G(x, eps + rdiff) - G(x, eps)) / rdiff

# Calculates the tangent to the path at the current point as (Gx^{-1} G_eps, -1).
def computeTangent(Gx_v, G_eps, prev_tangent, M, tolerance):
    x0 = prev_tangent[0:M] / prev_tangent[M]
    A = slg.LinearOperator(matvec=Gx_v, shape=(M,M))

    _tangent = slg.gmres(A, G_eps, x0=x0, atol=tolerance)[0]
    tangent = np.append(_tangent, -1.0)
    tangent = tangent / lg.norm(tangent)

    if np.dot(tangent, prev_tangent) > 0:
        return tangent
    else:
        return -tangent

"""
The Internal Numerical Continuation Routine.
"""
def numericalContinuation(x0, eps0, initial_tangent, M, max_steps, ds, ds_min, ds_max, tolerance):
    x = np.copy(x0)
    eps = eps0
    prev_tangent = np.copy(initial_tangent)

    x_path = [np.copy(x)]
    eps_path = [eps]
    print_str = 'Step {0:3d}:\t <u>: {1:4f}\t eps: {2:4f}\t ds: {3:6f}'.format(0, np.mean(x_path[0][0:N]), eps, ds)
    print(print_str)

    for n in range(1, max_steps+1):
		# Calculate the tangent to the curve at current point 
        Gx_v = lambda v: dGdx_v(x, v, eps)
        Geps = dGdeps(x, eps)
        tangent = computeTangent(Gx_v, Geps, prev_tangent, M, tolerance)

		# Create the extended system for corrector: z = (x, eps) = (u, v, eps)
        N_opt = lambda z: np.dot(tangent, z - np.append(x, eps)) + ds
        F = lambda z: np.append(G(z[0:M], z[M]), N_opt(z))

		# Our implementation uses adaptive timetepping
        while ds > ds_min:
			# Predictor: Extrapolation
            x_p = x + ds * tangent[0:M]
            eps_p = eps + ds * tangent[M]
            z_p = np.append(x_p, eps_p)

			# Corrector: Newton - Krylov
            try:
                z_new = opt.newton_krylov(F, z_p, f_tol=tolerance)
                x = z_new[0:M]
                eps = z_new[M]
                x_path.append(np.copy(x))
                eps_path.append(eps)

				# Updating the arclength step and tangent vector
                ds = min(1.2*ds, ds_max)
                prev_tangent = np.copy(tangent)

                break
            except:
                # Decrease arclength if the corrector fails.
                ds = max(0.5*ds, ds_min)
        else:
            print('Minimal Arclength Size is too large. Aborting.')
            return x_path, eps_path
		
        print_str = 'Step {0:3d}:\t <u>: {1:4f}\t eps: {2:4f}\t ds: {3:6f}'.format(n, np.mean(x_path[-1][0:N]), eps, ds)
        print(print_str)

    return x_path, eps_path

"""
Routine that calculates the bifurcation diagram of the DeepONet for the Fitzhugh-Nagumo PDE.
"""
def calculateBifurcationDiagram():
    # Load the initial condition from file.
    eps0 = 0.1
    x0 = np.load('./Results/DeepONet_steadystate_eps=' + str(eps0).replace('.', 'p') + '.npy')

    # Continuation Parameters
    M = 2 * N
    tolerance = 1.e-6
    max_steps = 1900
    ds_min = 1.e-6
    ds_max = 0.01
    ds = 0.001

    # Calculate the tangent to the path at the initial condition x0
    rng = rd.RandomState()
    random_tangent = rng.normal(0.0, 1.0, M+1)
    initial_tangent = computeTangent(lambda v: dGdx_v(x0, v, eps0), dGdeps(x0, eps0), random_tangent / lg.norm(random_tangent), M, tolerance)
    initial_tangent = initial_tangent / lg.norm(initial_tangent)

    # Do actual numerical continuation in both directions
    if initial_tangent[-1] < 0.0: # Decreasing eps
        print('Increasing eps first')
        sign = 1.0
    else:
        sign = -1.0
    x1_path, eps1_path = numericalContinuation(x0, eps0,  sign * initial_tangent, M, max_steps, ds, ds_min, ds_max, tolerance)
    x2_path, eps2_path = numericalContinuation(x0, eps0, -sign * initial_tangent, M, max_steps, ds, ds_min, ds_max, tolerance)

    # Store the full path
    x1_path = np.array(x1_path)
    x2_path = np.array(x2_path)
    eps1_path = np.array(eps1_path)
    eps2_path = np.array(eps2_path)
    np.save('./Results/bf_diagram.npy', np.hstack((x1_path, eps1_path[:,np.newaxis], x2_path, eps2_path[:,np.newaxis])))

    # Plot both branches
    plot_x1_path = np.average(x1_path[:, 0:N], axis=1)
    plot_x2_path = np.average(x2_path[:, 0:N], axis=1)
    plt.plot(eps1_path, plot_x1_path, color='blue')
    plt.plot(eps2_path, plot_x2_path, color='blue')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$<u>$')
    plt.show()



if __name__ == '__main__':
    calculateBifurcationDiagram()