import torch as pt
import numpy as np
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, device, dtype):
        n_initials = 100
        n_teeth = 10
        n_micro_points = 11
        n_data_rows_per_initial = 9995
        self.input_data = pt.zeros((n_initials * n_teeth * n_micro_points * n_data_rows_per_initial, 14), requires_grad=False, device=device, dtype=dtype)
        self.output_data = pt.zeros((n_initials * n_teeth * n_micro_points * n_data_rows_per_initial, 1), requires_grad=False, device=device, dtype=dtype)

        data_index = 0
        store_directory = './data/'
        for n in range(n_initials):
            for patch in range(n_teeth):
                data = np.load(store_directory + 'Initial='+str(n)+'_patch='+str(patch) + '.npy')
                n_time_steps = data.shape[0]-1

                branch_batch = pt.tensor(data[0:n_time_steps,:n_micro_points+2])
                for k in range(n_micro_points):
                    x_val = data[0,n_micro_points+2+k]
                    input_batch = pt.cat((branch_batch, x_val * pt.ones((branch_batch.shape[0], 1))), dim=1)
                    output_batch = pt.tensor(data[1:,k])[:,None]
                
                    self.input_data[data_index:data_index+n_data_rows_per_initial,:] = input_batch
                    self.output_data[data_index:data_index+n_data_rows_per_initial,:] = output_batch
                    data_index += n_data_rows_per_initial

        print((self.input_data.numel()*self.input_data.element_size() + self.output_data.numel()*self.output_data.element_size()) / 1024.0**2, 'MB')

    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, index):
        return (self.input_data[index,:], self.output_data[index,:])

if __name__ == '__main__':
    dataset = PatchDataset(pt.device('cpu'), pt.float32)