import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from functools import partial
import jax.nn as jnn
from sklearn import metrics

import os
from scipy.io import loadmat
import scipy.io as io
import pickle

import sys
sys.path.append("../..")

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9})

import warnings
warnings.filterwarnings("ignore")

# Check where gpu is enable or not
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


# Load the data
data = loadmat('../data/multiparameter/data_FHN.mat')

# Convert NumPy arrays to PyTorch tensors
inputs1_train = jnp.array(data['U1_train_branch'])[:,0:200]/21
inputs2_train = jnp.array(data['U2_train_branch'])[:,0:200]/1e4
input_param = jnp.array(data['U2_train_branch'])[:,200:201]
inputs_train = jnp.concat([inputs1_train,inputs2_train,input_param], axis = -1)
outputs1_train = jnp.array(data['V1_train_out'])/21
outputs2_train = jnp.array(data['V2_train_out'])/1e4

inputs1_test = jnp.array(data['U1_test_branch'])[:,0:200]/21
inputs2_test = jnp.array(data['U2_test_branch'])[:,0:200]/1e4
input_param = jnp.array(data['U2_test_branch'])[:,200:201]
inputs_test = jnp.concat([inputs1_test,inputs2_test,input_param], axis = -1)

outputs1_test = jnp.array(data['V1_test_out'])/21
outputs2_test = jnp.array(data['V2_test_out'])/1e4

grid = jnp.array(data['Ytrain_trunk'])

# Check the shapes of the subsets
print("grid:", grid.shape)
print("Shape of inputs_train:", inputs_train.shape)
print("Shape of outputs1_train:", outputs1_train.shape)
print("Shape of outputs1_test:", outputs1_test.shape)
print("Shape of inputs_train:", inputs_train.shape)
print("Shape of inputs_test:", inputs_test.shape)
print("Shape of outputs2_train:", outputs2_train.shape)
print("Shape of outputs2_test:", outputs2_test.shape)
print('#'*100)


def BranchNet(params, x):
    """
    Implement the branch network of the DeepONet.

    Args:
    params (list): List of weight and bias tuples for each layer.
    x (array): Input to the branch network.

    Returns:
    array: Output of the branch network.
    """
    def single_forward(params, x):
        for w, b in params:
            outputs = jnp.dot(x, w) + b
            x = jnn.silu(outputs)
        return outputs

    return vmap(partial(single_forward, params))(x)

def TrunkNet(params, x):
    """
    Implement the trunk network of the DeepONet.

    Args:
    params (list): List of weight and bias tuples for each layer.
    x (float): First input to the trunk network.
    t (float): Second input to the trunk network.

    Returns:
    array: Output of the trunk network.
    """
    inputs = jnp.array(x)
    for w, b in params:
        outputs = jnp.dot(x, w) + b
        x = jnn.silu(outputs)
    return outputs

@jit
def DeepONet(params, branch_inputs, trunk_inputs):
    """
    Implement the complete DeepONet architecture.

    Args:
    params (tuple): Tuple containing branch and trunk network parameters.
    branch_inputs (array): Inputs for the branch network.
    trunk_inputs (array): Inputs for the trunk network.

    Returns:
    array: Output of the DeepONet.
    """
    params_branch, params_trunk = params
    branch_outputs = lambda x: BranchNet(params_branch, x)
    b_out = branch_outputs(branch_inputs)
    trunk_output = lambda y: TrunkNet(params_trunk, y)
    t_out = trunk_output(trunk_inputs)
    results1 = jnp.einsum('ik, lk -> il',b_out[:,0:p], t_out[:,0:p])
    results2 = jnp.einsum('ik, lk -> il',b_out[:,p:2*p], t_out[:,p:2*p])
    return results1, results2

# network parameters.
p = 300 # Number of output neurons in both the branch and trunk net outputs.
nx = 401
input_neurons_branch = nx # m
input_neurons_trunk = 1

layer_sizes_b = [input_neurons_branch] + [100]*6 + [2*p]
layer_sizes_t = [input_neurons_trunk] + [100]*6 + [2*p]


bs = 100000 #batch size
iteration_list, loss_list, test_loss_list = [], [], []
iteration = 0

n_epochs = 100000
num_samples = len(inputs_train)

def save_model_params(params, resultdir, filename='model_params.pkl'):
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    save_path = os.path.join(resultdir, filename)
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_model_params(resultdir, filename='model_params.pkl'):
    load_path = os.path.join(resultdir, filename)
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return params

# params_branch, params_trunk = params
# Load the best model parameters
resultdir = './Results/'
best_params = load_model_params(resultdir, filename='model_params_best.pkl')
print("Loaded best model parameters")

# Predictions
mse_list = []

branch_inputs = inputs_test
print(branch_inputs.shape)

trunk_inputs = grid
prediction1, prediction2 = DeepONet(best_params, branch_inputs, trunk_inputs) # (bs, neval) 
input_test1 = inputs_test[:,0:200]*21
input_test2 = inputs_test[:,200:400]*1e4
parameter = inputs_test[:,400:401]
save_dict = {'u1_test': input_test1, 'u2_test': input_test2, 'eps': parameter, 'v1_pred': prediction1, 'v2_pred': prediction2, \
             'v1_target': outputs1_test, 'v2_target': outputs2_test}

io.savemat(resultdir+'/pred.mat', save_dict)
# Plotting all the results on -1 to 1 scale for better understanding
for i in range(inputs_test.shape[0]):

    branch_inputs = inputs_test[i].reshape(1, nx) 
    trunk_inputs = grid # (neval, 1) 
    
    prediction1_i, prediction2_i = DeepONet(best_params, branch_inputs, trunk_inputs) # (bs, neval)
    target1_i = outputs1_test[i]
    target2_i = outputs2_test[i]
    mse_i = np.mean((prediction1_i - target1_i)**2) + np.mean((prediction2_i - target2_i)**2)
    mse_list.append(mse_i.item())
  
    if i % 500 == 0:
        print('TEST SAMPLE '+str(i+1))

        r2score = metrics.r2_score(target1_i.flatten(), prediction1_i.flatten())
        relerror = np.linalg.norm(target1_i- prediction1_i) / np.linalg.norm(target1_i)
        r2score = float('%.4f'%r2score)
        relerror = float('%.4f'%relerror)
        print('Component-1 Rel. L2 Error = '+str(relerror)+', R2 score = '+str(r2score))

        r2score = metrics.r2_score(target2_i.flatten(), prediction2_i.flatten())
        relerror = np.linalg.norm(target2_i- prediction2_i) / np.linalg.norm(target2_i)
        r2score = float('%.4f'%r2score)
        relerror = float('%.4f'%relerror)
        print('Component-2 Rel. L2 Error = '+str(relerror)+', R2 score = '+str(r2score))

        fig = plt.figure(figsize=(15, 4))

        # Adjust subplot parameters for better spacing
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)

        # Input plot
        ax = fig.add_subplot(1, 4, 1)
        plt.plot(inputs_test[i,0:200], color='blue', linewidth=2)
        plt.plot(inputs_test[i,200:400], color='red', linewidth=2)
        plt.title('Inputs', fontsize=14)
        plt.legend(['Component 1', 'Component 2'])

        # Output plot
        ax = fig.add_subplot(1, 4, 2)
        target1 = target1_i.reshape(grid.shape[0])
        prediction1 = prediction1_i.reshape(grid.shape[0])
        plt.plot(target1, color='blue', linewidth=2)
        plt.plot(prediction1, color='green', linewidth=2, linestyle = 'dotted')
        plt.title('Component:1 Output Field', fontsize=14)
        plt.legend(['Target', 'Prediction'])

        ax = fig.add_subplot(1, 4, 3)
        target2 = target2_i.reshape(grid.shape[0])
        prediction2 = prediction2_i.reshape(grid.shape[0])
        plt.plot(target2, color='red', linewidth=2)
        plt.plot(prediction2, color='green', linewidth=2, linestyle = 'dotted')
        plt.title('Component:2 Output Field', fontsize=14)
        plt.legend(['Target', 'Prediction'])

        # Error plot
        ax = fig.add_subplot(1, 4, 4)
        error1 = target1 - prediction1
        error2 = target2 - prediction2
        plt.plot(error1, color='blue', linewidth=2)
        plt.plot(error2, color='red', linewidth=2)
        #plt.yscale("log")  
        plt.legend(['Component 1', 'Component 2'])
        plt.title('Absolute Error', fontsize=14)

        print('#'*230)


mse = sum(mse_list) / len(mse_list)
print("Mean Squared Error Test :\n", mse)

plt.show()