import numpy as np
import scipy.optimize as opt
import scipy.sparse.linalg as slg

def rhs(u, dx, params):
    u_left = np.roll(u, -1)
    u_right = np.roll(u, 1)
    u_xx = (u_left - 2.0*u + u_right) / dx**2

    u_rhs = u_xx + params['lambda'] * np.exp(u)
    u_rhs[0] = 0.0
    u_rhs[-1] = 0.0
    return u_rhs

def timestepper(u, dx, dt, T, params, verbose=False):
    if np.abs(u[0]) > 1.e-15 or np.abs(u[-1]) > 1.e-15:
        u[0] = 0.0
        u[-1] = 0.0

    n_steps = int(T / dt)
    for n in range(n_steps):
        if verbose:
            print('t =', (n+1) * dt)

        u = u + dt * rhs(u, dx, params)
        u[0] = 0.0
        u[-1] = 0.0 # Dirichlet boundary conditions
    
    return u

def psi(u, T_psi, dx, dt, params):
    return u - timestepper(u, dx, dt, T_psi, params)

def calculateSteadyState(u0, T_psi, dx, dt, params, verbose=False):
    F = lambda x: psi(x, T_psi, dx, dt, params)
    ss = opt.newton_krylov(F, u0, f_tol=1.e-14, verbose=verbose)
    return ss

def calculateRHSEigenvalues(x_ss, dx, params, k=10):
    M = x_ss.size
    
    r_diff = 1.e-8
    d_f_mvp = lambda w: (rhs(x_ss + r_diff * w, dx, params) - rhs(x_ss, dx, params)) / r_diff
    Df = slg.LinearOperator(shape=(M, M), matvec=d_f_mvp)
    f_eigvals, f_eigvecs = slg.eigs(Df, k=k, which='SM', return_eigenvectors=True)

    return f_eigvals, f_eigvecs

def calculateLeadingEigenvalues(x_ss, T_psi, dx, dt, params, k=10):
    M = x_ss.size
    
    r_diff = 1.e-8
    d_psi_mvp = lambda w: (psi(x_ss + r_diff * w, T_psi, dx, dt, params) - psi(x_ss, T_psi, dx, dt, params)) / r_diff
    D_psi = slg.LinearOperator(shape=(M, M), matvec=d_psi_mvp)
    psi_eigvals, psi_eigvecs = slg.eigs(D_psi, k=k, which='SM', return_eigenvectors=True)

    sorted_indices = np.argsort(psi_eigvals)
    return psi_eigvals[sorted_indices], psi_eigvecs[:,sorted_indices]