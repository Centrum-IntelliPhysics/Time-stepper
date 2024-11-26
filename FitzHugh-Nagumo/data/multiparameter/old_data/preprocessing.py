#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import sys
import scipy.io as io

directory = "./"
data_matrix = []

for filename in sorted(os.listdir(directory)):
    if filename.startswith("FHN_Evolution_eps=") and filename.endswith(".npy"):
        file_path = os.path.join(directory, filename)
        data = np.load(file_path)
        data_matrix.append(data)

eps = np.array([-0.01,0.0,0.005,0.01,0.013,0.015,0.017,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.92]).reshape(-1,1)
eps = np.tile(eps,[1,20]).reshape([400,1,1])
eps = np.tile(eps,[1,448,1])

data_matrix = np.array(data_matrix)
del data
data = np.zeros([data_matrix.shape[0]*data_matrix.shape[1],data_matrix.shape[-2],data_matrix.shape[-1]])

for iBranch in range(data_matrix.shape[0]):
    init_samp = data_matrix[iBranch,:,:,:]
    data[20*iBranch:20*(iBranch+1),:,:] = init_samp
del data_matrix

U1_branch = data[:,0:-1,0:200]
U1_branch = np.concatenate([U1_branch,eps], axis = -1)
U1_branch = np.reshape(U1_branch, [U1_branch.shape[0]*U1_branch.shape[1],201])

V1_out = data[:,1::,0:200]
V1_out = np.reshape(V1_out, [V1_out.shape[0]*V1_out.shape[1],200])

U2_branch = data[:,0:-1,200:400]
U2_branch = np.concatenate([U2_branch,eps], axis = -1)
U2_branch = np.reshape(U2_branch, [U2_branch.shape[0]*U2_branch.shape[1],201])

V2_out = data[:,1::,200:400]
V2_out = np.reshape(V2_out, [V2_out.shape[0]*V2_out.shape[1],200])

index = np.random.permutation(U1_branch.shape[0])
num_train = 174200
num_test = 5000
index_train = index[0:num_train]
index_test = index[-num_test:]

U1_train_branch = U1_branch[index_train]
U2_train_branch = U2_branch[index_train]

U1_test_branch = U1_branch[index_test]
U2_test_branch = U2_branch[index_test]

V1_train_out = V1_out[index_train]
V2_train_out = V2_out[index_train]

V1_test_out = V1_out[index_test]
V2_test_out = V2_out[index_test]

grid = np.reshape(np.linspace(0,1,200),[200,1])
save_dict = {'U1_train_branch': U1_train_branch, 'U2_train_branch': U2_train_branch,\
             'V1_train_out': V1_train_out, 'V2_train_out': V2_train_out,\
             'U1_test_branch': U1_test_branch, 'U2_test_branch': U2_test_branch,\
             'V1_test_out': V1_test_out, 'V2_test_out': V2_test_out, 'Ytrain_trunk': grid}

io.savemat('./data_FHN.mat', save_dict)

