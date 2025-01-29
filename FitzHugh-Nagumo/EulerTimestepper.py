import numpy as np
import scipy.optimize as opt
import scipy.sparse.linalg as slg

def fhn_rhs(u, v, dx, params):
    u_left = np.roll(u, -1)
    u_right = np.roll(u, 1)
    v_left = np.roll(v, -1)
    v_right = np.roll(v, 1)

    u_xx = (u_left - 2.0*u + u_right) / dx**2
    v_xx = (v_left - 2.0*v + v_right) / dx**2

    u_rhs = u_xx + u - u**3 - v
    v_rhs = params['delta'] * v_xx + params['eps'] * (u - params['a1']*v - params['a0'])

    return u_rhs, v_rhs

def fhn_euler(u, v, dx, dt, params):
    u_rhs, v_rhs = fhn_rhs(u, v, dx, params)
    u_new = u + dt * u_rhs
    v_new = v + dt * v_rhs

    # Apply homogeneous Neumann boundary conditions
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]
    v_new[0] = v_new[1]
    v_new[-1] = v_new[-2]

    return u_new, v_new

def fhn_euler_timestepper(u, v, dx, dt, T, params, verbose=False):
    N_steps = int(T / dt)
    for k in range(N_steps):
        if verbose:
            print('t =', k * dt)
        u, v = fhn_euler(u, v, dx, dt, params)
    return u, v

def psi(x, T, dx, dt, params):
    N = x.size // 2
    u, v = x[0:N], x[N:]

    u_new, v_new = fhn_euler_timestepper(u, v, dx, dt, T, params)
    return np.concatenate((u - u_new, v - v_new))

def calculateSteadyState(x0, T_psi, dx, dt, params):
    F = lambda x: psi(x, T_psi, dx, dt, params)
    ss = opt.newton_krylov(F, x0, f_tol=1.e-14)
    return ss

def calculateLeadingEigenvalues(x_ss, T_psi, dx, dt, params, k=10):
    M = x_ss.size
    
    r_diff = 1.e-8
    d_psi_mvp = lambda w: T_psi * (psi(x_ss + r_diff * w, T_psi, dx, dt, params) - psi(x_ss, T_psi, dx, dt, params)) / r_diff
    D_psi = slg.LinearOperator(shape=(M, M), matvec=d_psi_mvp)
    psi_eigvals, psi_eigvecs = slg.eigs(D_psi, k=k, which='SM', return_eigenvectors=True)

    sorted_indices = np.argsort(psi_eigvals)
    return psi_eigvals[sorted_indices], psi_eigvecs[:,sorted_indices]
