import torch as pt
import numpy as np

import os
from torch.utils.data import Dataset

# Sanity Settings
pt.set_default_dtype(pt.float64)

# The data impleementation and loader class
class DeepONetDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.total_data_rows = 18 * 200 * 20 * 48
        grid_size = 200
        grid = pt.linspace(0.0, 1.0, grid_size)
        input_data_tensor = pt.zeros((self.total_data_rows, 2 * grid_size + 2), requires_grad=False) # (u(t), v(t), eps, x_i)
        output_data_tensor = pt.zeros((self.total_data_rows, 2), requires_grad=False) # (u(x_i, t+1), v(x_i, t+1))
        
        data_index = 0
        directory = './../data/multiparameter/'
        for file in os.listdir(directory):
            if not file.startswith('FHN_BF_Evolution'):
                continue

            # Load all data in the file
            eps = float(file[file.find('eps=')+4:file.find('.')].replace('p', '.'))
            file_data = pt.from_numpy(np.load(directory + file))
            print('eps =', eps)

            for initial_index in range(file_data.shape[0]):
                for time_index in range(file_data.shape[1]-1):
                    u_t = file_data[initial_index, time_index, 0:200]
                    v_t = file_data[initial_index, time_index, 200:]
                    u_tp1 = file_data[initial_index, time_index+1, 0:200]
                    v_tp1 = file_data[initial_index, time_index+1, 200:]

                    partial_input_row = pt.concatenate((u_t, v_t, pt.tensor([eps])))
                    input_block = pt.tile(partial_input_row, dims=(grid_size, 1))
                    input_block = pt.concatenate((input_block, grid[:,None]), dim=1)
                    input_data_tensor[data_index:data_index+200,:] = input_block

                    output_block = pt.concatenate((u_tp1[:,None], v_tp1[:,None]), dim=1)
                    output_data_tensor[data_index:data_index+200,:] = output_block

                    data_index += 200

        # Downsampling. Check ratio # datapoints / # parameters!
        print('Downsampling...')
        self.data_size = 2 * 262144
        indices = pt.randperm(self.total_data_rows)[0:self.data_size]
        self.input_data = input_data_tensor[indices,:]
        self.output_data = output_data_tensor[indices,:]
        self.input_data.requires_grad = False
        self.output_data.requires_grad = False
        print('Total Data Memory Consumption:', (self.input_data.nelement() * self.input_data.element_size() + self.output_data.nelement() * self.output_data.element_size()) / 1000000.0, 'MB')

    def __len__(self):
        return self.data_size
	
    def __getitem__(self, idx):
        return self.input_data[idx,:], self.output_data[idx,:]
    
class SingleEpsilonDeepONetDataset(Dataset):
    def __init__(self):
        super().__init__()

        n_initials = 20
        n_time_points = 2000
        grid_size = 200
        self.total_data_rows = n_initials * n_time_points * grid_size

        grid = pt.linspace(0.0, 1.0, grid_size)
        input_data_tensor = pt.zeros((self.total_data_rows, 2 * grid_size + 1), requires_grad=False) # (u(t), v(t), x_i)
        output_data_tensor = pt.zeros((self.total_data_rows, 2), requires_grad=False) # (u(x_i, t+1), v(x_i, t+1))
        
        data_index = 0

        # Load all data in the file
        directory = './../data/multiparameter/'
        for initial_index in range(20):
            file = 'FHN_BF_Evolution_Initial=' + str(initial_index) + '_eps=0p1_dT=0p001.npy'
            file_data = pt.from_numpy(np.load(directory + file))

            for time_index in range(file_data.shape[0]-1):
                x_t = file_data[time_index, :]
                u_tp1 = file_data[time_index+1,0:200]
                v_tp1 = file_data[time_index+1, 200:]

                input_block = pt.tile(x_t, dims=(grid_size, 1))
                input_block = pt.concatenate((input_block, grid[:,None]), dim=1)
                input_data_tensor[data_index:data_index + grid_size,:] = input_block

                output_block = pt.concatenate((u_tp1[:,None], v_tp1[:,None]), dim=1)
                output_data_tensor[data_index:data_index + grid_size,:] = output_block

                data_index += grid_size

        # Downsampling. Check ratio # datapoints / # parameters!
        print('Downsampling...')
        self.data_size = 262144
        indices = pt.randperm(self.total_data_rows)[0:self.data_size]
        self.input_data = input_data_tensor[indices,:]
        self.output_data = output_data_tensor[indices,:]
        self.input_data.requires_grad = False
        self.output_data.requires_grad = False
        print('Total Data Memory Consumption:', (self.input_data.nelement() * self.input_data.element_size() + self.output_data.nelement() * self.output_data.element_size()) / 1024.0**2, 'MB')

    def __len__(self):
        return self.data_size
	
    def __getitem__(self, idx):
        return self.input_data[idx,:], self.output_data[idx,:]
    
if __name__ == '__main__':
    dataset = SingleEpsilonDeepONetDataset()

    import time
    time.sleep(120)