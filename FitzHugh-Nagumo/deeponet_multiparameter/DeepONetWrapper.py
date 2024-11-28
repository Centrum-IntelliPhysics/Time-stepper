import os
import pickle

import jax.numpy as np
import jax.nn as jnn

# The DeepONet class for the Fitzhugh-Nagumo equation represenst a map from one state (u, v, eps) to another (u, v)
# one second later.
class DeepONetWrapper:
    def __init__(self):
        resultdir = './Results/'
        self.best_params = self.load_model_params(resultdir, filename='model_params_best.pkl')
        self.best_branch_params = self.best_params[0]
        self.best_trunk_params = self.best_params[1]
        self.p = 300

    def load_model_params(self, resultdir, filename='model_params.pkl'):
        load_path = os.path.join(resultdir, filename)
        with open(load_path, 'rb') as f:
            params = pickle.load(f)
        return params
    
    def BranchNet(self, x):
        for w, b in self.best_branch_params:
            outputs = np.dot(x, w) + b
            x = jnn.silu(outputs)
        return outputs

        #return vmap(partial(single_forward, self.best_branch_params))(x)

    def TrunkNet(self, x):
        for w, b in self.best_trunk_params:
            outputs = np.dot(x, w) + b
            x = jnn.silu(outputs)
        return outputs

    def DeepONet(self, u, v, eps, grid):
        branch_inputs = np.concatenate((u, v, np.array([eps])))[np.newaxis,:]
        trunk_inputs = grid[:,np.newaxis] 

        b_out = self.BranchNet(branch_inputs)
        t_out = self.TrunkNet(trunk_inputs)
        u = np.einsum('ik, lk -> il', b_out[:,0:self.p], t_out[:,0:self.p])
        v = np.einsum('ik, lk -> il', b_out[:,self.p:2*self.p], t_out[:,self.p:2*self.p])

        u = u[0,:]
        v = v[0,:]
        return u, v
    
    def __call__(self, u, v, eps, grid):
        return self.DeepONet(u, v, eps, grid)