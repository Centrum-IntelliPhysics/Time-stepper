
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import os
import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import sys
from Adam import Adam
import scipy
torch.manual_seed(0)
np.random.seed(0)

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        #x_ft = torch.rfft(x,1,normalized=True,onesided=False)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
    

        self.fc1 = nn.Linear(self.width, 32)
        self.fc2 = nn.Linear(32, 1) 

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        #x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        #x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

################################################################
#  configurations
################################################################
save_index = 1

batch_size_train = 900
batch_size_test = 200
learning_rate = 0.001

epochs = 300
step_size = 50
gamma = 0.5

modes = 50
width = 32

################################################################
# load data and data normalization
################################################################
reader = MatReader('../data/singleparameter/data_single_parameter_022_AC_new_aligned_healing_shorter2')
x_train = reader.read_field('Utrain_branch').T
x_init = torch.reshape(x_train[:,0:100],(-1,100,1))
x_parameter = torch.reshape(torch.tile(x_train[:,100::],[1,100]),(-1,100,1))
x_train = torch.concat((x_init,x_parameter),axis=-1 )
y_train = reader.read_field('Vtrain_out').T
y_train = torch.reshape(y_train,(-1,100,1))

x_test = reader.read_field('Utest_branch').T
x_init = torch.reshape(x_test[:,0:100],(-1,100,1))
x_parameter = torch.reshape(torch.tile(x_test[:,100::],[1,100]),(-1,100,1))
x_test = torch.concat((x_init,x_parameter),axis=-1 )
y_test = reader.read_field('Vtest_out').T
y_test = torch.reshape(y_test,(-1,100,1))

ntrain = y_train.shape[0]
ntest = y_test.shape[0]

x_normalizer = GaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = GaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)
y_test = y_normalizer.encode(y_test)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size_test, shuffle=True)

# model
model = FNO1d(modes, width) #.cuda()

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()

train_error = np.zeros((epochs, 1))
test_error = np.zeros((epochs, 1))

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    for x, y in train_loader:
        x, y = x, y

        optimizer.zero_grad()
        out = model(x)

        loss = F.mse_loss(out, y)
        loss.backward()
        y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)
        #l2 = myloss(out, y)
        optimizer.step()
        train_mse += loss.item()

    scheduler.step()
    model.eval()
    test_mse = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x, y 

            out = model(x)
            
            y = y_normalizer.decode(y)
            out = y_normalizer.decode(out)            
            test_mse += F.mse_loss(out, y).item()

    train_mse /= len(train_loader)
    test_mse /= ntest

    train_error[ep,0] = train_mse
    test_error[ep,0] = test_mse
    t2 = default_timer()
    print("Epoch: %d, time: %.3f, Train Loss: %.3e, Test Loss: %.4f" % (ep, t2-t1, train_mse, test_mse))

elapsed = time.time() - start_time
print("\n=============================")
print("Training done...")
print('Training time: %.3f'%(elapsed))
print("=============================\n")

# ====================================
# saving settings
# ====================================
current_directory = os.getcwd()
case = "Results"
folder_index = str(save_index)

results_dir = "/" + case +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/test_error.txt', test_error)    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
torch.save(model, save_models_to+'AC_singleparameter')

################################################################
# testing
################################################################
batch_size_test = 1
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                          batch_size = batch_size_test, shuffle=False)    
pred = torch.zeros(y_test.shape)

index = 0
error = 0
t1 = default_timer()
dataSegment = "Test"
with torch.no_grad():
    for x, y in test_loader:
        
        x, y = x, y
        out = model(x)
        y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)            
        pred[index,:,:] = out   
        error += F.mse_loss(out, y).item()       
        index = index + 1

error = error/index
t2 = default_timer()
testing_time = t2-t1

x_test = x_normalizer.decode(x_test)
y_test = y_normalizer.decode(y_test)
scipy.io.savemat(save_results_to+'AC_singleparameter_test.mat', 
                  mdict={'x_test': x_test.detach().cpu().numpy(),
                         'y_test': y_test.numpy(), 
                         'y_pred': pred.cpu().numpy(),
                         'test_error': error})  

print("\n=============================")
print('Testing error: %.3e'%(error))
print("=============================\n")    

# Plotting the loss history
num_epoch = epochs
x = np.linspace(1, num_epoch, num_epoch)
fig = plt.figure(constrained_layout=False, figsize=(7, 7))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(x, train_error[:,0], color='blue', label='Training Loss')
ax.plot(x, test_error[:,0], color='red', label='Testing Loss')
ax.set_yscale('log')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
ax.legend(loc='upper left')
fig.savefig(save_results_to+'loss_history.png')
