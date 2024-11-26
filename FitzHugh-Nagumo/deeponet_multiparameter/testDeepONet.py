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
    eps = 0.1

    # Initial condition
    x_array = np.linspace(0.0, L, N) / L
    u = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v = sigmoid(x_array, 10, 0.0, 2.0, 0.1)

    deeponet = DeepONetWrapper()

    # Timestepping
    dt = 1.0
    T = 100.0
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_array, u, label='u(x, t=0.0)')
    ax2.plot(x_array, v, label='v(x, t=0.0)')
    for n in range(int(T / dt)):
        u, v = deeponet.DeepONet(u, v, eps, x_array)
        print(u.shape)

        ax1.plot(x_array, u, label='u(x, t='+str(n*dt)+')')
        ax2.plot(x_array, v, label='v(x, t='+str(n*dt)+')')

    # Plotting the final result
    ax1.set_title(r'$u(x,t)$')
    ax2.set_title(r'$v(x,t)$')
    ax1.legend()
    plt.show()

if __name__ == '__main__':
    plotFitzHughNagumoSolution()