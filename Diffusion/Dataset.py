import torch as pt
import numpy as np

from torch.utils.data import Dataset
    
class DiffusionDataset(Dataset):
    def __init__(self, device, dtype):
        super().__init__()

        n_initials = 1000
        n_datarows_per_initial = 2**14
        grid_size = 50
        self.total_data_rows = n_initials * n_datarows_per_initial

        grid = pt.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
        self.input_data = pt.zeros((self.total_data_rows, grid_size + 1), requires_grad=False, device=device, dtype=dtype) # (u(t), x_i)
        self.output_data = pt.zeros((self.total_data_rows, 1), requires_grad=False, device=device, dtype=dtype) # u(x_i, t+1)
        
        data_index = 0

        # Load all data in the file
        directory = './data/'
        for initial_index in range(n_initials):
            print('Reading file', initial_index+1)

            file = 'trajectory_' + str(initial_index) + '.npy'
            file_data = pt.from_numpy(np.load(directory + file))
            file_data = file_data.to(dtype=dtype, device=device)

            # Go through the file and put all timesteps in a large tensor
            n_time_points = file_data.shape[0]
            input_time_block = pt.zeros(((n_time_points-1) * grid_size, grid_size + 1))
            output_time_block = pt.zeros(((n_time_points-1) * grid_size, 1))
            for time_index in range(n_time_points-1):
                u_t = file_data[time_index, :]
                u_tp1 = file_data[time_index+1, :]

                input_block = pt.tile(u_t, dims=(grid_size, 1))
                input_block = pt.concatenate((input_block, grid[:,None]), dim=1)
                output_block = u_tp1[:,None]
                input_time_block[time_index * grid_size:(time_index+1)*grid_size,:] = input_block
                output_time_block[time_index * grid_size:(time_index+1)*grid_size,:] = output_block

            # Downsample to the eps_initial time tensor to 2**12 rows
            indices = pt.randperm(input_time_block.shape[0])[0:n_datarows_per_initial]
            self.input_data[data_index:data_index + n_datarows_per_initial,:] = input_time_block[indices,:]
            self.output_data[data_index:data_index + n_datarows_per_initial,:] = output_time_block[indices,:]
        
            data_index += n_datarows_per_initial

        # Downsampling. Check ratio # datapoints / # parameters!
        self.input_data.requires_grad = False
        self.output_data.requires_grad = False
        print('Total Data Memory Consumption:', (self.input_data.nelement() * self.input_data.element_size() + self.output_data.nelement() * self.output_data.element_size()) / 1024.0**2, 'MB')

    def __len__(self):
        return self.total_data_rows
	
    def __getitem__(self, idx):
        return self.input_data[idx,:], self.output_data[idx,:]
    
if __name__ == '__main__':
    dataset = DiffusionDataset(device="cpu", dtype=pt.float64)

    import time
    time.sleep(120)