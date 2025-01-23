import numpy as np
import scipy.optimize as opt

def pde_rhs(u, dx, params):
    u_left = np.roll(u, -1)
    u_right = np.roll(u, 1)

    u_xx = (u_left - 2.0*u + u_right) / dx**2
    u_rhs = params['D'] * u_xx

    # Apply Dirichlet boundary conditions
    u_rhs[0] = 0.0
    u_rhs[-1] = 0.0

    return u_rhs

def euler(u, dx, dt, params):
    u_rhs = pde_rhs(u, dx, params)
    u_new = u + dt * u_rhs

    return u_new

def euler_timestepper(u, dx, dt, T, params, verbose=False):
    N_steps = int(T / dt)
    for k in range(N_steps):
        if verbose:
            print('t =', k * dt)
        u = euler(u, dx, dt, params)
    return u

def psi(u, T, dx, dt, params):
    u_new = euler_timestepper(u, dx, dt, T, params)
    return u - u_new

def calculateSteadyState(u0, T_psi, dx, dt, params):
    F = lambda u: psi(u, T_psi, dx, dt, params)
    ss = opt.newton_krylov(F, u0, f_tol=1.e-14)

    return ss