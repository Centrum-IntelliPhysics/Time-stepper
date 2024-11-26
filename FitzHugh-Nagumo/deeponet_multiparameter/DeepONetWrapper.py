import os
import pickle
from functools import partial

import jax.numpy as jnp
import jax.nn as jnn
from jax import jit, vmap

class DeepONetWrapper:
    def __init__(self):
        resultdir = './Results/'
        self.best_params = self.load_model_params(resultdir, filename='model_params_best.pkl')

        self.p = 300 # Number of output neurons in both the branch and trunk net outputs.
        self.nx = 401

    def load_model_params(self, resultdir, filename='model_params.pkl'):
        load_path = os.path.join(resultdir, filename)
        with open(load_path, 'rb') as f:
            params = pickle.load(f)
        return params
    
    def BranchNet(self, params, x):
        def single_forward(params, x):
            for w, b in params:
                outputs = jnp.dot(x, w) + b
                x = jnn.silu(outputs)
            return outputs

        return vmap(partial(single_forward, params))(x)

    def TrunkNet(self, params, x):
        inputs = jnp.array(x)
        for w, b in params:
            outputs = jnp.dot(x, w) + b
            x = jnn.silu(outputs)
        return outputs

    @jit
    def DeepONet(self, params, branch_inputs, trunk_inputs):
        # Branch_inputs and trunk_inputs are 1D arrays. I convert then to (1, .) and (., 1) shape here
        branch_inputs = branch_inputs[jnp.newaxis,:]
        trunk_inputs = trunk_inputs[:,jnp.newaxis]
        branch_inputs[0, 0:200] = branch_inputs[0, 0:200] / 21 # Reschaling u(x) ?
        branch_inputs[0,200:400] = branch_inputs[0, 200:400] / 1.e4 # Reschaling v(x) ? 

        params_branch, params_trunk = params
        b_out = self.BranchNet(params_branch, branch_inputs)
        t_out = self.TrunkNet(params_trunk, trunk_inputs)
        results1 = jnp.einsum('ik, lk -> il',b_out[:,0:self.p], t_out[:,0:self.p])
        results2 = jnp.einsum('ik, lk -> il',b_out[:,self.p:2*self.p], t_out[:,self.p:2*self.p])
        return results1, results2
    
    def __call__(self, params, branch_inputs, trunk_inputs):
        return self.DeepONet(params, branch_inputs, trunk_inputs)