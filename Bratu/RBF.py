import numpy as np
import scipy.linalg as sclg

class RBFInterpolator:
    lu_exists = False
    lu_piv = None

    @staticmethod
    def createSystem(x, n, kernel): # n is the number of points
        X, Y = np.meshgrid(x, x)
        A = kernel(X, Y)
        RBFInterpolator.lu_piv = sclg.lu_factor(A)
        RBFInterpolator.lu_exists = True

    def __init__(self, x, f, solver='lu_direct'):
        self.x = np.copy(x)
        self.f = np.copy(f)
        self.n = len(self.x)

        self.sigma = 2.0 / self.n
        self.kernel = lambda x, y: np.exp(-0.5 * (x - y)**2 / self.sigma**2)

        if solver == 'lu_direct' and RBFInterpolator.lu_exists is False:
            RBFInterpolator.createSystem(self.x, self.n, self.kernel)
        self.w = sclg.lu_solve(RBFInterpolator.lu_piv, self.f)
        self.functional = lambda y: np.sum(np.array([self.w[i] * self.kernel(y, self.x[i]) for i in range(self.n)]))
        self.d_functional = lambda y: np.sum(np.array([-self.w[i] * (y - self.x[i]) / self.sigma**2 * self.kernel(y, self.x[i]) for i in range(self.n)]))

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        val_array = np.zeros_like(x)
        for i in range(len(x)):
            val_array[i] = self.functional(x[i])
        return val_array
    
    def derivative(self, x):
        return self.d_functional(x)

