import torch as pt
import numpy as np
import os

from torch.utils.data import Dataset
    
class MultipleEpsilonDeepONetDataset(Dataset):
    def __init__(self, device, dtype, from_file=False):
        super().__init__()

        directory = './../data/multiparameter/'
        if from_file:
            self.input_data = pt.Tensor(np.load(directory + 'inputs.npy'), dtype=dtype, device=device)
            self.output_data = pt.Tensor(np.load(directory + 'outputs.npy'), dtype=dtype, device=device)
            self.input_data.requires_grad = False
            self.output_data.requires_grad = False
            self.total_data_rows = self.input_data.shape[0]

            return

        eps_values = self.loadEpsValues(directory)[0:19]
        print(eps_values)
        n_eps = len(eps_values)
        n_initials_per_eps = 200
        n_datarows_per_initial = 2**10
        grid_size = 200
        self.total_data_rows = n_eps * n_initials_per_eps * n_datarows_per_initial

        grid = pt.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
        self.input_data = pt.zeros((self.total_data_rows, 1 + 2 * grid_size + 1), requires_grad=False, device=device, dtype=dtype) # (eps, u(t), v(t), x_i)
        self.output_data = pt.zeros((self.total_data_rows, 2), requires_grad=False, device=device, dtype=dtype) # (u(x_i, t+1), v(x_i, t+1))
        data_index = 0

        # Load all data in the file
        for eps_index in range(n_eps):
            eps = eps_values[eps_index]
            eps_grid = eps * pt.ones((grid_size, 1), device=device, dtype=dtype)
            print('Reading epsilon', eps)

            for initial_index in range(n_initials_per_eps):
                print('Reading file', initial_index)
                file = 'FHN_MultiEps_Evolution_Initial=' + str(initial_index) + '_eps=' + str(eps).replace('.', 'p') + '.npy'
                file_data = pt.from_numpy(np.load(directory + file))
                file_data = file_data.to(dtype=dtype, device=device)

                # Go through the file and put all timesteps in a large tensor
                n_time_points = file_data.shape[0]
                input_time_block = pt.zeros(((n_time_points-1) * grid_size, 1 + 2 * grid_size + 1))
                output_time_block = pt.zeros(((n_time_points-1) * grid_size, 2))
                for time_index in range(n_time_points-1):
                    x_t = file_data[time_index, :]
                    u_tp1 = file_data[time_index+1,0:200]
                    v_tp1 = file_data[time_index+1, 200:]

                    input_block = pt.tile(x_t, dims=(grid_size, 1))
                    input_block = pt.concatenate((eps_grid, input_block, grid[:,None]), dim=1)
                    output_block = pt.concatenate((u_tp1[:,None], v_tp1[:,None]), dim=1)
                    input_time_block[time_index * grid_size:(time_index+1)*grid_size,:] = input_block
                    output_time_block[time_index * grid_size:(time_index+1)*grid_size,:] = output_block

                # Downsample to the eps_initial time tensor to 2**10 rows
                indices = pt.randperm(input_time_block.shape[0])[0:n_datarows_per_initial]
                self.input_data[data_index:data_index + n_datarows_per_initial,:] = input_time_block[indices,:]
                self.output_data[data_index:data_index + n_datarows_per_initial,:] = output_time_block[indices,:]
            
                data_index += n_datarows_per_initial

        # Downsampling. Check ratio # datapoints / # parameters!
        self.input_data.requires_grad = False
        self.output_data.requires_grad = False
        print('Total Data Memory Consumption:', (self.input_data.nelement() * self.input_data.element_size() + self.output_data.nelement() * self.output_data.element_size()) / 1024.0**2, 'MB')

        np.save(directory + 'inputs.npy', self.input_data.cpu().numpy())
        np.save(directory + 'outputs.npy', self.output_data.cpu().numpy())

    def loadEpsValues(self, directory):
            eps_values = []
            file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            for file in file_names:
                if not file.startswith('FHN_MultiEps_Evolution_Initial='):
                    continue
                eps_index = file.find('eps=')
                end_index = file.find('.npy')
                eps_values.append(float(file[eps_index+4:end_index].replace('p', '.')))
            eps_values = list(set(eps_values))
            eps_values.sort()
            return eps_values
   
    def __len__(self):
        return self.total_data_rows
	
    def __getitem__(self, idx):
        return self.input_data[idx,:], self.output_data[idx,:]
    
if __name__ == '__main__':
    dataset = MultipleEpsilonDeepONetDataset(device="cpu", dtype=pt.float32, from_file=False)

    import time
    time.sleep(120)