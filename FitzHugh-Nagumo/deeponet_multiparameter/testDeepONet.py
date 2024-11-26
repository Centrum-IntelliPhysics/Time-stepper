import jax.numpy as np
import jax.numpy.linalg as lg
import matplotlib.pyplot as plt

from DeepONetWrapper import DeepONetWrapper


def sigmoid(x_array, x_center=0.0, y_center=0.0, x_scale=1.0, y_scale=1.0):
    return y_scale / (1.0 + np.exp(-(x_array  - x_center)/x_scale)) + y_center

def plotFitzHughNagumoSolution():
    # Model parameters
    L = 20.0
    N = 200
    dx = L / N
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1 # 0.01 originally for the spatio-temporal oscillations
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Initial condition
    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 10, 0.0, 2.0, 0.1)
    u = np.copy(u0)
    v = np.copy(v0)

    # Timestepping
    dt = 1.e-3
    report_dt = 10*dt
    T = 450.0
    u_solution = np.zeros((int(T / report_dt)+1, u.size))
    v_solution = np.zeros((int(T / report_dt)+1, u.size))
    u_solution[0,:] = u0
    v_solution[0,:] = v0
    for n in range(int(T / dt)):
        u, v = fhn_euler(u, v, dx, dt, params)

        if n > 0 and n % 10 == 0:
            u_solution[n // 10, :] = u
            v_solution[n // 10, :] = v

    # Store the time-evolution steady - state
    np.save(directory + 'euler_evolution_T=' + str(T) + '.npy', np.vstack((x_array, u, v)))

    # Plotting the final result
    x_plot_array = np.linspace(0.0, T, u_solution.shape[1]+1)
    t_plot_array = np.linspace(0.0, T, u_solution.shape[0]+1)
    plt.plot(x_array, u, label='u(x, t=450)')
    plt.plot(x_array, v, label='v(x, t=450)')
    plt.legend()

    X, Y = np.meshgrid(x_plot_array, t_plot_array)
    v_solution = sigmoid(v_solution, y_scale=2.0, y_center=-1.0, x_scale=0.05)
    u_max = np.max(u_solution)
    u_min = np.min(u_solution)
    v_max = np.max(v_solution)
    v_min = np.min(v_solution)
    print('psi', lg.norm(psi(np.concatenate((u,v)), 0.1, dx, dt, params)))
    plt.figure()
    plt.pcolor(X, Y, u_solution, cmap='viridis', vmin=min(u_min, v_min), vmax=max(u_max, v_max))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title(r'$u(x, t)$')
    plt.figure()
    plt.pcolor(X, Y, v_solution, cmap='viridis', vmin=min(u_min, v_min), vmax=max(u_max, v_max))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title(r'$v(x, t)$')
    plt.show()